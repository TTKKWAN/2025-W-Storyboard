#!/bin/bash

# GPU 설정 (여러 GPU 사용)
GPU_IDS=(6 7)
NUM_SHARDS=${#GPU_IDS[@]}

# [경로 설정]
BASE_MODEL="/home/aikusrv01/storyboard/TK/model_sd"
PROMPT_FILE="/home/aikusrv01/storyboard/TK/validation/validation.jsonl"
BATCH_SCRIPT="/home/aikusrv01/storyboard/TK/validation/batch_generate.py"
OUTPUT_ROOT="/home/aikusrv01/storyboard/TK/validation/output"

시드
SEEDS=(9999)

CHECKPOINTS=(
  "/home/aikusrv01/storyboard/TK/model_weight/1125/train_trigger"
  "/home/aikusrv01/storyboard/TK/model_weight/1125/train_trigger_cluster"
)

run_generation () {
  local checkpoint_path="$1"
  local seed="$2"
  local checkpoint_name
  checkpoint_name="$(basename "$checkpoint_path")"

  local output_dir="$OUTPUT_ROOT/${checkpoint_name}"
  mkdir -p "$output_dir"

  echo "🧪 ${checkpoint_name} | seed ${seed} | 검증 프롬프트 이미지 생성 시작..."

  local pids=()
  for idx in "${!GPU_IDS[@]}"; do
    local gpu_id="${GPU_IDS[$idx]}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    python "$BATCH_SCRIPT" \
      --jsonl "$PROMPT_FILE" \
      --output-dir "$output_dir" \
        --base-model "$BASE_MODEL" \
        --checkpoint "$checkpoint_path" \
        --lora-scale 1.0 \
        --seed "$seed" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$idx" \
        --fuse-lora
    ) &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  echo "✅ ${checkpoint_name} | seed ${seed} 완료"
}

for checkpoint_path in "${CHECKPOINTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_generation "$checkpoint_path" "$seed"
  done
done

echo "🎯 모든 체크포인트/시드 조합 완료. 결과는 $OUTPUT_ROOT에 저장되었습니다."
