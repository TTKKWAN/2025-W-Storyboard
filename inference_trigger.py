import argparse
from pathlib import Path
import torch
import os

from safetensors.torch import load_file

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stable Diffusion with fine-tuned LoRA and Custom Tokens.")
    
    parser.add_argument("--base-model", default="/home/aikusrv01/storyboard/TK/stable_diffusion", help="기본 모델 경로")
    parser.add_argument("--checkpoint", required=True, help="학습된 체크포인트 폴더 경로")
    parser.add_argument("--weight-name", default="pytorch_lora_weights.safetensors", help="LoRA 파일명")
    parser.add_argument("--trigger-word", default=None, help="프롬프트 맨 앞에 추가할 트리거 워드")
    parser.add_argument("--prompt", default=None, required=True, help="프롬프트")
    parser.add_argument("--negative-prompt", default="low quality, bad anatomy, worst quality, text, watermark, blurry, ugly", help="부정 프롬프트")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA 강도")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default="/home/aikusrv01/storyboard/TK/img_output/sample.png")
    parser.add_argument("--fuse-lora", action="store_true")
    parser.add_argument("--allow-downloads", action="store_false", dest="local_files_only")
    parser.set_defaults(local_files_only=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    # output_path.parent.TKdir(parents=True, exist_ok=True)

    print(f"🔹 Loading Base Model from: {args.base_model}")

    # ========================================================================
    # [수정 핵심] 체크포인트 폴더 구조에 맞춰 유연하게 로딩
    # ========================================================================
    tokenizer = None
    text_encoder = None
    
    # 1. 토크나이저 로드 시도 (폴더 루트에 added_tokens.json이 있는지 확인)
    try:
        print(f"🔍 Looking for custom tokenizer in: {checkpoint_path}")
        # subfolder=None으로 설정하여 체크포인트 루트에서 직접 찾게 함
        tokenizer = CLIPTokenizer.from_pretrained(args.checkpoint, subfolder=None, local_files_only=args.local_files_only)
        print("✅ Custom Tokenizer loaded successfully from checkpoint root!")
    except Exception as e:
        print(f"⚠️ Failed to load tokenizer from root: {e}")
        print("ℹ️ Fallback: Using Base Model tokenizer.")
        tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
# ========================================================================
    # [수정됨] 2. 텍스트 인코더 강제 로드 (SafeTensors 직접 주입)
    # ========================================================================
    from safetensors.torch import load_file # 상단 import에 추가 필요하지만 여기서 호출해도 됨

    print(f"🔍 Loading Base Text Encoder first...")
    # 1) 일단 베이스 모델의 텍스트 인코더를 불러옵니다.
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder", torch_dtype=torch.float16)
    
    # 2) 토크나이저 크기에 맞춰 사이즈를 늘립니다 (이걸 해야 가중치를 넣을 수 있음)
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    # 3) 체크포인트에 있는 model.safetensors (학습된 텍스트 인코더 가중치)를 덮어씌웁니다.
    custom_weights_path = checkpoint_path / "model.safetensors"
    
    if custom_weights_path.exists():
        print(f"♻️ Overwriting Text Encoder weights from: {custom_weights_path}")
        try:
            # safetensors 파일 로드
            state_dict = load_file(str(custom_weights_path))
            
            # 가중치 강제 주입 (strict=False로 해서 형식이 조금 달라도 중요 부분만 로드)
            missing, unexpected = text_encoder.load_state_dict(state_dict, strict=False)
            print("✅ Custom Text Encoder weights loaded successfully!")
        except Exception as e:
            print(f"⚠️ Failed to load custom weights manually: {e}")
            print("ℹ️ Running with random initialization for trigger word (Effect will be weak).")
    else:
        print("⚠️ model.safetensors not found in checkpoint. Trigger word might not work.")

    # 파이프라인 생성
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        torch_dtype=torch.float16,
        local_files_only=args.local_files_only,
        safety_checker=None,
        requires_safety_checker=False
    ).to("cuda")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    print(f"🔹 Loading LoRA weights from: {args.checkpoint}")
    try:
        pipe.load_lora_weights(args.checkpoint, weight_name=args.weight_name)
        if args.fuse_lora:
            pipe.fuse_lora(lora_scale=args.lora_scale)
            print(f"✅ LoRA Fused with scale: {args.lora_scale}")
    except Exception as e:
        print(f"❌ Error loading LoRA: {e}")
        return

    generator = torch.Generator(device="cuda")
    if args.seed is not None:
        generator.manual_seed(args.seed)

    # 프롬프트 조합
    final_prompt = args.prompt
    if args.trigger_word:
        if final_prompt:
            final_prompt = f"{args.trigger_word}, {final_prompt}"
        else:
            final_prompt = args.trigger_word
            
    print(f"📝 Final Prompt: {final_prompt}")
    
    result = pipe(
        prompt=final_prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        cross_attention_kwargs={"scale": args.lora_scale} if not args.fuse_lora else None,
        generator=generator,
    )
    
    image = result.images[0]
    image.save(output_path)
    print(f"💾 Saved Image to: {output_path}")

if __name__ == "__main__":
    main()