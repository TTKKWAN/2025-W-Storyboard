#!/usr/bin/env python
# coding=utf-8
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA and New Tokens."""

import argparse
import logging
import math
import os
import random
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True)
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--caption_column", type=str, default="text")
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_epochs", type=int, default=1)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="sd-model-finetuned-lora")
    parser.add_argument(
        "--run_metadata_file",
        type=str,
        default=None,
        help="Optional path to a text file where CLI args and periodic loss will be stored.",
    )
    parser.add_argument(
        "--loss_report_steps",
        type=int,
        default=50,
        help="Interval (in optimization steps) to append averaged loss into run_metadata_file.",
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", default=False, action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--snr_gamma", type=float, default=None)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--prediction_type", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--noise_offset", type=float, default=0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--image_interpolation_mode", type=str, default="lanczos")
    
    # [추가] 텍스트 인코더 학습 여부 명시적 인자
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args

DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}

def main():
    args = parse_args()
    
    # [중요] 여기에 학습시킬 트리거 워드를 정의합니다.
    # 이 단어들은 Tokenizer에 새로 추가됩니다.
    NEW_TOKENS = ['<ms_trg>', '<cu_trg>', '<fs_trg>']
    
    # [중요] 초기화 맵핑: 새로운 토큰이 빨리 학습되도록 비슷한 의미의 기존 단어로 초기화합니다.
    TOKEN_INITIALIZER = {
        '<ms_trg>': 'medium shot', # medium shot과 비슷하게 시작
        '<cu_trg>': 'close-up shot',  # close up과 비슷하게 시작
        '<fs_trg>': 'full shot',    # full shot과 비슷하게 시작
        '<ls_trg>': 'long shot'
    }

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process and args.run_metadata_file is not None:
        metadata_path = Path(args.run_metadata_file)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(metadata_path, "w") as metadata_file:
                metadata_file.write("\n".join(sys.argv[1:]))
                metadata_file.write("\n\n")
        except OSError as exc:
            logger.warning(f"Could not write metadata file {metadata_path}: {exc}")

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    # ================= [1. 토큰 추가 및 초기화 로직] =================
    num_added_tokens = tokenizer.add_tokens(NEW_TOKENS)

    if num_added_tokens > 0:
        logger.info(f"✅ Added {num_added_tokens} new tokens to the tokenizer.")
        # 토큰이 추가되었으므로 임베딩 크기 조절
        text_encoder.resize_token_embeddings(len(tokenizer))
        
        # [스마트 초기화] 랜덤 대신 기존 단어의 임베딩을 복사하여 학습 속도 부스팅
        logger.info("🔄 Initializing new tokens with existing embeddings for faster convergence...")
        token_embeds = text_encoder.get_input_embeddings().weight.data
        
        for new_token, source_word in TOKEN_INITIALIZER.items():
            if new_token in NEW_TOKENS:
                # 새로운 토큰의 ID 찾기
                new_id = tokenizer.convert_tokens_to_ids(new_token)
                # 소스 단어(예: medium)의 ID 찾기
                src_ids = tokenizer.encode(source_word, add_special_tokens=False)
                if len(src_ids) > 0:
                    src_id = src_ids[0]
                    # 가중치 복사
                    token_embeds[new_id] = token_embeds[src_id].clone()
                    logger.info(f"   Initialized {new_token} from '{source_word}'")

    # ================= [1. 완료] =================

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    
    # Freeze everything first
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 텍스트 인코더 학습 설정 (입력 임베딩 레이어만)
    if num_added_tokens > 0 or args.train_text_encoder:
        text_encoder.get_input_embeddings().weight.requires_grad_(True)
        text_encoder.get_input_embeddings().to(dtype=torch.float32)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # LoRA Config
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=torch.float32) # Text Encoder는 학습 위해 fp32 유지 권장

    unet.add_adapter(unet_lora_config)
    if args.mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            logger.warning("xformers is not available.")

    # ================= [2. Optimizer 설정] =================
    
    params_to_optimize = []
    
    # UNet LoRA 파라미터
    lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if len(lora_layers) > 0:
        params_to_optimize.append({"params": lora_layers})
        logger.info(f"✅ UNet LoRA Params: {len(lora_layers)} layers included.")

    # Text Encoder 임베딩 파라미터 (트리거 워드 학습용)
    embedding_params = list(text_encoder.get_input_embeddings().parameters())
    embedding_trainable_params = [p for p in embedding_params if p.requires_grad]
    
    if len(embedding_trainable_params) > 0:
        # 임베딩 학습은 좀 더 높은 학습률이 필요할 수 있어 Multiplier 적용
        params_to_optimize.append({
            "params": embedding_trainable_params,
            "lr": args.learning_rate * 5.0 
        })
        logger.info(f"✅ Text Encoder Embeddings: Included for training new tokens.")
    else:
        logger.warning("⚠️ Warning: Text Encoder embeddings are NOT being trained. Trigger words will NOT work.")

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes")
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # ================= [2. 완료] =================

    # Dataset loading
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, data_dir=args.train_data_dir)
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset("imagefolder", data_files=data_files, cache_dir=args.cache_dir)

    column_names = dataset["train"].column_names
    image_column = args.image_column if args.image_column in column_names else column_names[0]
    caption_column = args.caption_column if args.caption_column in column_names else column_names[1]

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError("Caption column format error")
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    train_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers
    )

    len_train_dataloader = len(train_dataloader)
    len_train_dataloader_after_sharding = math.ceil(len_train_dataloader / accelerator.num_processes)
    num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps_for_scheduler,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train Loop
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    # Resume logic omitted for brevity (standard logic)
    
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    should_log_loss_to_file = (
        args.run_metadata_file is not None and args.loss_report_steps is not None and args.loss_report_steps > 0
    )
    loss_window_sum = 0.0
    loss_window_count = 0

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Latents
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Noise
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
                
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Text Embedding (Gradients flow here for NEW TOKENS!)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    params_to_clip = list(unet.parameters())
                    if num_added_tokens > 0:
                        params_to_clip += list(text_encoder.get_input_embeddings().parameters())
                    
                    params_to_clip = [p for p in params_to_clip if p.requires_grad]
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)

                if should_log_loss_to_file and accelerator.is_main_process:
                    loss_window_sum += train_loss
                    loss_window_count += 1
                    if global_step % args.loss_report_steps == 0 and loss_window_count > 0:
                        avg_loss = loss_window_sum / loss_window_count
                        with open(args.run_metadata_file, "a") as metadata_file:
                            metadata_file.write(f"step {global_step}: {avg_loss:.6f}\n")
                        loss_window_sum = 0.0
                        loss_window_count = 0
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        
                        # 1. UNet LoRA 저장
                        accelerator.save_state(save_path)
                        unwrapped_unet = unwrap_model(unet)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )
                        
                        # 2. [중요] Tokenizer 및 TextEncoder 저장
                        # 새로운 토큰이 추가되었으므로, 이를 포함한 tokenizer와 text_encoder를 통째로 저장해야 나중에 쓸 수 있음
                        tokenizer.save_pretrained(save_path)
                        unwrap_model(text_encoder).save_pretrained(save_path)
                        
                        logger.info(f"Saved state, LoRA weights, Tokenizer, and TextEncoder to {save_path}")

            if global_step >= args.max_train_steps:
                break

    # Final Save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )
        
        # [중요] 최종 결과에도 Tokenizer와 Text Encoder 저장
        tokenizer.save_pretrained(args.output_dir)
        unwrap_model(text_encoder).save_pretrained(args.output_dir)
        logger.info("✅ Final Save: Tokenizer and Text Encoder saved successfully.")

    accelerator.end_training()

if __name__ == "__main__":
    main()
