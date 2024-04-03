deepspeed --include localhost:4,5,6,7 ha_dpo/models/llava-v1_5/train_dpo_post.py \
    --lora_enable False \
    --deepspeed ha_dpo/models/llava-v1_5/scripts/zero3.json \
    --model_name_or_path /home/cuiruochen/model/llava-v1.5-7b \
    --version v1 \
    --vg_path ha_dpo/data/VG \
    --desc_data_path ha_dpo/data/hadpo/llava-v1.5/desc_data.json \
    --pope_data_path ha_dpo/data/hadpo/llava-v1.5/pope_data.json \
    --vision_tower /home/cuiruochen/model/clip-vit-large-patch14-336 \
    --freeze_backbone True \
    --tune_mm_mlp_adapter False \
    --tune_lm_head True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ha_dpo/models/llava-v1_5/checkpoints/llava-post-decoder-bs-1-1-16-epoch-3-gpu-4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "llava-v1.5" \
    --beta 0.1