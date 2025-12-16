CUDA_VISIBLE_DEVICES=4,5 accelerate launch --main_process_port 29506 --num_processes=2 /home/aikusrv01/storyboard/github_ver/train.py \
  --pretrained_model_name_or_path="/home/aikusrv01/storyboard/github_ver/model_sd" \
  --train_data_dir="/home/aikusrv01/storyboard/github_ver/Dataset_fin" \
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

  
  
  