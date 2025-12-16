#!/bin/bash

# GPU 설정
export CUDA_VISIBLE_DEVICES=6

# [경로 설정] - 아까 성공한 경로로 유지
CHECKPOINT_PATH="/home/aikusrv01/storyboard/TK/sd_weight/1212/train_fin"
OUTPUT_DIR="/home/aikusrv01/storyboard/TK/model_output/fin/tigger_cluster_aug"

BASE_MODEL="/home/aikusrv01/storyboard/TK/model_sd"
INFERENCE_SCRIPT="/home/aikusrv01/storyboard/TK/train_triger/inference_trigger.py"

# 고정된 프롬프트 (샷 정보 제외)
COMMON_PROMPT="Eye level, female, youth, happy, slim body, white shirt, black pants, no background, day time"


SEED=9999  # 시드를 고정해야 정확한 비교가 가능함

mkdir -p "$OUTPUT_DIR"

echo "🧪 트리거 워드 성능 실험 시작..."

# ------------------------------------------------------------------------------
# [실험 1] 
# ------------------------------------------------------------------------------

python "$INFERENCE_SCRIPT" \
  --base-model "$BASE_MODEL" \
  --checkpoint "$CHECKPOINT_PATH" \
  --trigger-word "<ms_trg>" \
  --prompt "medium shot, $COMMON_PROMPT" \
  --lora-scale 1.0 \
  --seed $SEED \
  --fuse-lora \
  --output "$OUTPUT_DIR/ms_shot.png"


python "$INFERENCE_SCRIPT" \
  --base-model "$BASE_MODEL" \
  --checkpoint "$CHECKPOINT_PATH" \
  --trigger-word "<cu_trg>" \
  --prompt "close-up shot, $COMMON_PROMPT" \
  --lora-scale 1.0 \
  --seed $SEED \
  --fuse-lora \
  --output "$OUTPUT_DIR/cu_shot.png"


python "$INFERENCE_SCRIPT" \
  --base-model "$BASE_MODEL" \
  --checkpoint "$CHECKPOINT_PATH" \
  --trigger-word "<fs_trg>" \
  --prompt "full shot, $COMMON_PROMPT" \
  --lora-scale 1.0 \
  --seed $SEED \
  --fuse-lora \
  --output "$OUTPUT_DIR/fs_shot.png"

