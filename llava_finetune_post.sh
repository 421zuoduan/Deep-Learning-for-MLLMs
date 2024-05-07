version1='v8'
time1='20240507'
train_bs1='2'
eval_bs1='1'
gradient_accumulation_steps1='16'
epoch1='1'
localhost='0,1,2,3'
gpu1='4'
lr1='5e-7'

source_folder="/home/cuiruochen/model/llava-v1.5-7b-----------pib-to-be-copied"
target_folder1="/home/cuiruochen/model/llava-v1.5-7b-post-${time1}-${version1}-bs-${train_bs1}-${eval_bs1}-${gradient_accumulation_steps1}-epoch-${epoch1}-gpu-${gpu1}-lr-${lr1}-ablation-llava-post-sa"
output_dir1="post_interaction_block/models/llava-v1_5/checkpoints/llava-post-${time1}-${version1}-bs-${train_bs1}-${eval_bs1}-${gradient_accumulation_steps1}-epoch-${epoch1}-gpu-${gpu1}-lr-${lr1}-ablation-llava-post-sa"
# 检查目标文件夹路径是否正确
echo "target_folder：$target_folder1"

# 复制文件夹到目标路径, 如果已经存在路径, 会嵌套复制. 所以需要检查是否已经存在
if [ -d "$target_folder1" ]; then
    echo "Target folder already exists. Skipping copy."
else
    # 目标路径不存在，执行复制
    cp -r "$source_folder" "$target_folder1"

    # 检查复制是否成功
    if [ $? -eq 0 ]; then
        echo "Folder copied successfully!"
    else
        echo "Folder copy failed!"
        exit 1
    fi
fi
echo "Folders check completed"

deepspeed --include localhost:${localhost} --master_port 12145 post_interaction_block/models/llava-v1_5/train_llava_dpo_post.py \
    --lora_enable False \
    --deepspeed post_interaction_block/models/llava-v1_5/scripts/zero3.json \
    --model_name_or_path ${target_folder1} \
    --version v1 \
    --vg_path post_interaction_block/data/VG \
    --desc_data_path post_interaction_block/data/hadpo/llava-v1.5/desc_data.json \
    --pope_data_path post_interaction_block/data/hadpo/llava-v1.5/pope_data.json \
    --vision_tower /home/cuiruochen/model/clip-vit-large-patch14-336 \
    --freeze_backbone True \
    --tune_mm_mlp_adapter False \
    --tune_post_interaction_block True \
    --tune_lm_head False \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
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

# python post_interaction_block/models/llava-v1_5/replace_bin.py --tune_stage 1 \
#     --path_model_state_dict ${target_folder1}/pytorch_model-00002-of-00002.bin \
#     --path_non_lora_state_dict ${output_dir1}/non_lora_trainables.bin

# wait

# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu1} --master_port 12546 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
#     --coco_path post_interaction_block/data/coco2014 \
#     --pope_path post_interaction_block/data/POPE \
#     --model-path ${target_folder1} \
#     --set popular


# wait

# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu1} --master_port 12546 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
#     --coco_path post_interaction_block/data/coco2014 \
#     --pope_path post_interaction_block/data/POPE \
#     --model-path ${target_folder1} \
#     --set random

# wait

# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu1} --master_port 12546 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
#     --coco_path post_interaction_block/data/coco2014 \
#     --pope_path post_interaction_block/data/POPE \
#     --model-path ${target_folder1} \
#     --set adv

# wait



# version2='v8'
# time2='20240423'
# train_bs2='2'
# eval_bs2='1'
# gradient_accumulation_steps2='16'
# epoch2='2'
# localhost='4,5,6,7'
# gpu2='4'
# lr2='5e-7'

# source_folder="/home/cuiruochen/model/llava-v1.5-7b-----------to-be-copied"
# target_folder2="/home/cuiruochen/model/llava-v1.5-7b-train_post${version2}-${time2}-bs-${train_bs2}-${eval_bs2}-${gradient_accumulation_steps2}-epoch-${epoch2}-gpu-${gpu2}-lr-${lr2}"
# output_dir2="post_interaction_block/models/llava-v1_5/checkpoints/llava-post-decoder-${time2}-${version2}-bs-${train_bs2}-${eval_bs2}-${gradient_accumulation_steps2}-epoch-${epoch2}-gpu-${gpu2}-lr-${lr2}"

# # 检查目标文件夹路径是否正确
# echo "目标文件夹路径：$target_folder2"

# # 复制文件夹到目标路径, 如果已经存在路径, 会嵌套复制. 所以需要检查是否已经存在
# cp -r "$source_folder" "$target_folder2"

# # 检查复制是否成功
# if [ $? -eq 0 ]; then
#     echo "文件夹复制成功"
# else
#     echo "文件夹复制失败"
#     exit 1
# fi

# echo "操作完成"

# deepspeed --include localhost:${localhost} post_interaction_block/models/llava-v1_5/train_llava_dpo_post.py \
#     --lora_enable False \
#     --deepspeed post_interaction_block/models/llava-v1_5/scripts/zero3.json \
#     --model_name_or_path ${target_folder2} \
#     --version v1 \
#     --vg_path post_interaction_block/data/VG \
#     --desc_data_path post_interaction_block/data/hadpo/llava-v1.5/desc_data.json \
#     --pope_data_path post_interaction_block/data/hadpo/llava-v1.5/pope_data.json \
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
#     --output_dir ${output_dir2} \
#     --num_train_epochs ${epoch2} \
#     --per_device_train_batch_size ${train_bs2} \
#     --per_device_eval_batch_size ${eval_bs2} \
#     --gradient_accumulation_steps ${gradient_accumulation_steps2} \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate ${lr2} \
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


# python post_interaction_block/models/llava-v1_5/replace_bin.py --tune_stage 1 \
#     --path_model_state_dict ${target_folder2}/pytorch_model-00002-of-00002.bin \
#     --path_non_lora_state_dict ${output_dir2}/non_lora_trainables.bin

# wait





# version3='v8'
# time3='20240423'
# train_bs3='2'
# eval_bs3='1'
# gradient_accumulation_steps3='16'
# epoch3='1'
# localhost='0,1,2,3'
# gpu3='4'
# lr3='5e-7'

# source_folder="/home/cuiruochen/model/llava-v1.5-7b-----------to-be-copied"
# target_folder3="/home/cuiruochen/model/llava-v1.5-7b-train_post${version3}-${time3}-bs-${train_bs3}-${eval_bs3}-${gradient_accumulation_steps3}-epoch-${epoch3}-gpu-${gpu3}-lr-${lr3}"
# output_dir3="post_interaction_block/models/llava-v1_5/checkpoints/llava-post-decoder-${time3}-${version3}-bs-${train_bs3}-${eval_bs3}-${gradient_accumulation_steps3}-epoch-${epoch3}-gpu-${gpu3}-lr-${lr3}"

# # 检查目标文件夹路径是否正确
# echo "目标文件夹路径：$target_folder3"

# # 复制文件夹到目标路径, 如果已经存在路径, 会嵌套复制. 所以需要检查是否已经存在
# cp -r "$source_folder" "$target_folder3"

# # 检查复制是否成功
# if [ $? -eq 0 ]; then
#     echo "文件夹复制成功"
# else
#     echo "文件夹复制失败"
#     exit 1
# fi

# echo "操作完成"

# deepspeed --include localhost:${localhost} post_interaction_block/models/llava-v1_5/train_llava_dpo_post.py \
#     --lora_enable False \
#     --deepspeed post_interaction_block/models/llava-v1_5/scripts/zero3.json \
#     --model_name_or_path ${target_folder3} \
#     --version v1 \
#     --vg_path post_interaction_block/data/VG \
#     --desc_data_path post_interaction_block/data/hadpo/llava-v1.5/desc_data.json \
#     --pope_data_path post_interaction_block/data/hadpo/llava-v1.5/pope_data.json \
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
#     --output_dir ${output_dir3} \
#     --num_train_epochs ${epoch3} \
#     --per_device_train_batch_size ${train_bs3} \
#     --per_device_eval_batch_size ${eval_bs3} \
#     --gradient_accumulation_steps ${gradient_accumulation_steps3} \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate ${lr3} \
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


# python post_interaction_block/models/llava-v1_5/replace_bin.py --tune_stage 1 \
#     --path_model_state_dict ${target_folder3}/pytorch_model-00002-of-00002.bin \
#     --path_non_lora_state_dict ${output_dir3}/non_lora_trainables.bin

# wait

# version4='v8'
# time4='20240423'
# train_bs4='2'
# eval_bs4='1'
# gradient_accumulation_steps4='16'
# epoch4='2'
# localhost='0,1,2,3'
# gpu4='4'
# lr4='5e-7'

# source_folder="/home/cuiruochen/model/llava-v1.5-7b-----------to-be-copied"
# target_folder4="/home/cuiruochen/model/llava-v1.5-7b-train_post${version4}-${time4}-bs-${train_bs4}-${eval_bs4}-${gradient_accumulation_steps4}-epoch-${epoch4}-gpu-${gpu4}-lr-${lr4}"
# output_dir4="post_interaction_block/models/llava-v1_5/checkpoints/llava-post-decoder-${time4}-${version4}-bs-${train_bs4}-${eval_bs4}-${gradient_accumulation_steps4}-epoch-${epoch4}-gpu-${gpu4}-lr-${lr4}"

# # 检查目标文件夹路径是否正确
# echo "目标文件夹路径：$target_folder4"

# # 复制文件夹到目标路径, 如果已经存在路径, 会嵌套复制. 所以需要检查是否已经存在
# cp -r "$source_folder" "$target_folder4"

# # 检查复制是否成功
# if [ $? -eq 0 ]; then
#     echo "文件夹复制成功"
# else
#     echo "文件夹复制失败"
#     exit 1
# fi

# echo "操作完成"

# deepspeed --include localhost:${localhost} post_interaction_block/models/llava-v1_5/train_llava_dpo_post.py \
#     --lora_enable False \
#     --deepspeed post_interaction_block/models/llava-v1_5/scripts/zero3.json \
#     --model_name_or_path ${target_folder4} \
#     --version v1 \
#     --vg_path post_interaction_block/data/VG \
#     --desc_data_path post_interaction_block/data/hadpo/llava-v1.5/desc_data.json \
#     --pope_data_path post_interaction_block/data/hadpo/llava-v1.5/pope_data.json \
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
#     --output_dir ${output_dir4} \
#     --num_train_epochs ${epoch4} \
#     --per_device_train_batch_size ${train_bs4} \
#     --per_device_eval_batch_size ${eval_bs4} \
#     --gradient_accumulation_steps ${gradient_accumulation_steps4} \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate ${lr4} \
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


# python post_interaction_block/models/llava-v1_5/replace_bin.py --tune_stage 1 \
#     --path_model_state_dict ${target_folder4}/pytorch_model-00002-of-00002.bin \
#     --path_non_lora_state_dict ${output_dir4}/non_lora_trainables.bin


# wait



    

# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu1} --master_port 12546 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
#     --coco_path post_interaction_block/data/coco2014 \
#     --pope_path post_interaction_block/data/POPE \
#     --model-path ${target_folder1} \
#     --set popular


# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu1} --master_port $RANDOM post_interaction_block/models/llava-v1_5/pope_eval_post.py \
#     --coco_path post_interaction_block/data/coco2014 \
#     --pope_path post_interaction_block/data/POPE \
#     --model-path ${target_folder2} \
#     --set popular

wait

CUDA_VISIBLE_DEVICES=${localhost} python 哥们,留四张卡,跑大模型.py --size 30000 --gpus ${gpu1} --interval 0.01

