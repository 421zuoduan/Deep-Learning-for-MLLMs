# deepspeed --include localhost:0,1,3 ha_dpo/models/llava-v1_5/train_dpo_post.py \
#     --lora_enable False \
#     --deepspeed ha_dpo/models/llava-v1_5/scripts/zero3.json \
#     --model_name_or_path /home/cuiruochen/model/llava-v1.5-7b-train_postv7-20240418-bs-1-1-16-epoch-1-gpu-3-stages-2 \
#     --version v1 \
#     --vg_path ha_dpo/data/VG \
#     --desc_data_path ha_dpo/data/hadpo/llava-v1.5/desc_data.json \
#     --pope_data_path ha_dpo/data/hadpo/llava-v1.5/pope_data.json \
#     --vision_tower /home/cuiruochen/model/clip-vit-large-patch14-336 \
#     --freeze_backbone True \
#     --tune_mm_mlp_adapter False \
#     --tune_post_decoder True \
#     --tune_lm_head False \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ha_dpo/models/llava-v1_5/checkpoints/llava-post-decoder-20240418-v7-bs-1-1-16-epoch-1-gpu-3-stages-2/stage-1 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-4 \
#     --weight_decay 0. \
#     --warmup_steps 0 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --run_name "llava-v1.5" \
#     --beta 0.1


# python ha_dpo/models/llava-v1_5/replace_bin.py --tune_stage 1 \
#     --path_model_state_dict /home/cuiruochen/model/llava-v1.5-7b-train_postv7-20240418-bs-1-1-16-epoch-1-gpu-3-stages-2/pytorch_model-00002-of-00002.bin \
#     --path_non_lora_state_dict ha_dpo/models/llava-v1_5/checkpoints/llava-post-decoder-20240418-v7-bs-1-1-16-epoch-1-gpu-3-stages-2/stage-1/non_lora_trainables.bin 


CUDA_VISIBLE_DEVICES=0,1,3 torchrun --nproc_per_node 3 --master_port $RANDOM ha_dpo/models/llava-v1_5/pope_eval_post.py \
    --coco_path ha_dpo/data/coco2014 \
    --pope_path ha_dpo/data/POPE \
    --model-path /home/cuiruochen/model/llava-v1.5-7b-train_postv7-20240418-bs-1-1-16-epoch-1-gpu-3-stages-2 \
    --set adv