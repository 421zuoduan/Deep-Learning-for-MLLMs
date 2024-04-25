ersion1='v8'
time1='20240425'
train_bs1='4'
eval_bs1='1'
gradient_accumulation_steps1='8'
epoch1='1'
localhost='4,5,6,7'
gpu1='4'
lr1='5e-6'

source_folder="/home/cuiruochen/model/llava-v1.5-7b-----------to-be-copied"
target_folder1="/home/cuiruochen/model/llava-v1.5-7b-full-datasets-train_post${version1}-${time1}-bs-${train_bs1}-${eval_bs1}-${gradient_accumulation_steps1}-epoch-${epoch1}-gpu-${gpu1}-lr-${lr1}"
output_dir1="./checkpoints//llava-post-decoder-${time1}-${version1}-bs-${train_bs1}-${eval_bs1}-${gradient_accumulation_steps1}-epoch-${epoch1}-gpu-${gpu1}-lr-${lr1}"
# 检查目标文件夹路径是否正确
echo "目标文件夹路径：$target_folder1"

# 复制文件夹到目标路径, 如果已经存在路径, 会嵌套复制. 所以需要检查是否已经存在
source_folder="/home/cuiruochen/model/llava-v1.5-7b-----------to-be-copied"
cp -r "$source_folder" "$target_folder1"
# 检查复制是否成功
if [ $? -eq 0 ]; then
    echo "文件夹复制成功"
else
    echo "文件夹复制失败"
    exit 1
fi
echo "操作完成"

deepspeed --include localhost:${localhost} ha_dpo/models/llava-v1_5/train_llava_post.py \
    --deepspeed ha_dpo/models/llava-v1_5/scripts/zero3.json \
    --model_name_or_path ${target_folder1} \
    --version v1 \
    --data_path /home/cuiruochen/LLaVA/playground/data/llava_v1_5_mix665k.json \
    --image_folder /home/cuiruochen/LLaVA/playground/data \
    --vision_tower /home/cuiruochen/model/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_backbone True \
    --tune_post_decoder True \
    --freeze_mm_mlp_adapter True \
    --tune_mm_mlp_adapter False \
    --tune_llm_head False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${output_dir1} \
    --num_train_epochs ${epoch1} \
    --per_device_train_batch_size ${train_bs1} \
    --per_device_eval_batch_size ${eval_bs1} \
    --gradient_accumulation_steps ${gradient_accumulation_steps1} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate ${lr1} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

python ha_dpo/models/llava-v1_5/replace_bin.py --tune_stage 1 \
    --path_model_state_dict ${target_folder1}/pytorch_model-00002-of-00002.bin \
    --path_non_lora_state_dict ${output_dir1}/non_lora_trainables.bin

wait

CUDA_VISIBLE_DEVICES=${localhost} python 哥们,留四张卡,跑大模型.py --size 30000 --gpus 4 --interval 0.01
