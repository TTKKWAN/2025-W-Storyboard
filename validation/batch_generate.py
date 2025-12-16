import argparse
import json
import time
from pathlib import Path
import torch

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from safetensors.torch import load_file
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images for every entry in a JSONL file.")
    parser.add_argument("--jsonl", required=True, help="프롬프트(text)를 담고 있는 JSONL 경로")
    parser.add_argument("--output-dir", required=True, help="생성 이미지를 저장할 폴더")

    parser.add_argument("--base-model", required=True, help="기본 Stable Diffusion 모델 경로")
    parser.add_argument("--checkpoint", required=True, help="LoRA 체크포인트 경로")
    parser.add_argument("--weight-name", default="pytorch_lora_weights.safetensors")
    parser.add_argument("--negative-prompt", default="low quality, bad anatomy, worst quality, text, watermark, blurry, ugly")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=9999, help="베이스 시드 (샘플 index를 더해 사용)")
    parser.add_argument("--fuse-lora", action="store_true")
    parser.add_argument("--allow-downloads", action="store_false", dest="local_files_only")
    parser.add_argument("--max-samples", type=int, default=None, help="최대 생성 개수 (디버깅용)")
    parser.add_argument("--num-shards", type=int, default=1, help="병렬 처리를 위한 총 shard 수")
    parser.add_argument("--shard-index", type=int, default=0, help="이 프로세스가 처리할 shard index (0-based)")
    parser.set_defaults(local_files_only=True)
    return parser.parse_args()


def load_pipeline(args: argparse.Namespace) -> StableDiffusionPipeline:
    checkpoint_path = Path(args.checkpoint)

    print(f"🔹 Loading Base Model from: {args.base_model}")
    print(f"🔍 Loading tokenizer...")
    try:
        tokenizer = CLIPTokenizer.from_pretrained(args.checkpoint, subfolder=None, local_files_only=args.local_files_only)
        print("✅ Custom tokenizer loaded from checkpoint root")
    except Exception as exc:
        print(f"⚠️ Failed to load tokenizer from checkpoint root: {exc}")
        tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")

    print("🔍 Loading text encoder...")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder", torch_dtype=torch.float16)
    text_encoder.resize_token_embeddings(len(tokenizer))

    custom_weights_path = checkpoint_path / "model.safetensors"
    if custom_weights_path.exists():
        print(f"♻️ Loading text encoder weights from {custom_weights_path}")
        state_dict = load_file(str(custom_weights_path))
        text_encoder.load_state_dict(state_dict, strict=False)
    else:
        print("⚠️ model.safetensors not found. Trigger tokens might be weak.")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        torch_dtype=torch.float16,
        local_files_only=args.local_files_only,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    print(f"🔹 Loading LoRA from {args.checkpoint}")
    pipe.load_lora_weights(args.checkpoint, weight_name=args.weight_name)
    if args.fuse_lora:
        pipe.fuse_lora(lora_scale=args.lora_scale)
        print(f"✅ LoRA fused with scale {args.lora_scale}")

    return pipe


def load_prompts(jsonl_path: Path):
    entries = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            entry = json.loads(raw)
            prompt = entry.get("text", "").strip()
            if not prompt:
                print(f"⚠️ Empty prompt at line {line_no}, skipping.")
                continue
            file_name = entry.get("file_name", f"sample_{line_no}")
            entries.append((line_no - 1, file_name, prompt))
    return entries


def main() -> None:
    args = parse_args()

    if args.num_shards < 1:
        raise ValueError("num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard-index must be within [0, num-shards).")

    jsonl_path = Path(args.jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_entries = load_prompts(jsonl_path)
    shard_entries = [
        item for item in all_entries
        if item[0] % args.num_shards == args.shard_index
    ]

    pipe = load_pipeline(args)
    generator = torch.Generator(device="cuda")

    total = 0
    start_time = time.time()
    progress = tqdm(
        shard_entries,
        desc=f"Shard {args.shard_index+1}/{args.num_shards}",
        unit="img"
    )
    for idx, file_name, prompt in progress:
        if args.max_samples is not None and total >= args.max_samples:
            break

        if args.seed is not None:
            generator.manual_seed(args.seed + idx)

        final_prompt = prompt
        result = pipe(
            prompt=final_prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            cross_attention_kwargs={"scale": args.lora_scale} if not args.fuse_lora else None,
            generator=generator,
        )

        image = result.images[0]
        target_name = f"{Path(file_name).stem}.png"
        image_path = output_dir / target_name
        image.save(image_path)
        total += 1

    progress.close()
    elapsed = time.time() - start_time
    print(f"✅ Done. Generated {total} samples on shard {args.shard_index} in {elapsed/60:.2f} min ({elapsed:.1f} s).")


if __name__ == "__main__":
    main()
