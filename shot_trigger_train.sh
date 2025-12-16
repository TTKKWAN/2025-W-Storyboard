######## 원본 스크립트 파일 ##########
# accelerate launch train_text_to_image.py \
#   --pretrained_model_name_or_path="/home/aikusrv01/storyboard/TK/stable_diffusion" \
#   --train_data_dir="/home/aikusrv01/storyboard/TK/Training_test/01.원천데이터" \
#   --resolution=512 \
#   --train_batch_size=4 \
#   --num_train_epochs=50 \
#   --gradient_accumulation_steps=1 \
#   --output_dir="/home/aikusrv01/storyboard/TK/sd_weight"


######### shot_trigger_train.sh #########
######### Shot별로 Triggerword 설정하여 Train (수정버전) #########
CUDA_VISIBLE_DEVICES=4,5 accelerate launch --main_process_port 29506 --num_processes=2 /home/aikusrv01/storyboard/TK/train_triger/train_trigger02.py \
  --pretrained_model_name_or_path="/home/aikusrv01/storyboard/TK/model_sd" \
  --train_data_dir="/home/aikusrv01/storyboard/TK/Dataset_fin" \
  --image_column="image" \
  --caption_column="text" \
  --resolution=512 \
  --train_batch_size=6 \
  --num_train_epochs=25 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=400 \
  --checkpointing_steps=200 \
  --snr_gamma=5.0 \
  --rank=64 \
  --output_dir="/home/aikusrv01/storyboard/TK/sd_weight/1212/train_fin" \
  --run_metadata_file="/home/aikusrv01/storyboard/TK/sd_weight/1212/train_fin/run_metadata.txt" \
  --loss_report_steps=50 \
  --train_text_encoder \
  --mixed_precision="fp16"

  
  

# train_text_encoder 조건은 트리거 토큰을 새로 넣었을 때 텍스트 인코더의 임베딩을 함께 학습할지 여부를 결정하는 값
  