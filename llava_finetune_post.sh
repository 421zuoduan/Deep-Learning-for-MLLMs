localhost='0,1,2,3'
# localhost='4,5,6,7'

version1='v8'
time1='20240518'
train_bs1='2'
eval_bs1='1'
gradient_accumulation_steps1='16'
epoch1='1'
gpu1='4'
lr1='5e-7'

pope_file1="post_interaction_block/models/llava-v1_5/pope_adv.jsonl"
pope_file2="post_interaction_block/models/llava-v1_5/pope_random.jsonl"
pope_file3="post_interaction_block/models/llava-v1_5/pope_popular.jsonl"

source_folder="/home/cuiruochen/model/llava-v1.5-7b-----------pib-to-be-copied"
target_folder1="/home/cuiruochen/model/llava-v1.5-7b-post-${time1}-${version1}-bs-${train_bs1}-${eval_bs1}-${gradient_accumulation_steps1}-epoch-${epoch1}-gpu-${gpu1}-lr-${lr1}-gelu-align-linear"
output_dir1="post_interaction_block/models/llava-v1_5/checkpoints/llava-post-${time1}-${version1}-bs-${train_bs1}-${eval_bs1}-${gradient_accumulation_steps1}-epoch-${epoch1}-gpu-${gpu1}-lr-${lr1}-gelu-align-linear"
# 检查目标文件夹路径是否正确
echo "target_folder: $target_folder1"

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

deepspeed --include localhost:${localhost} --master_port 12156 post_interaction_block/models/llava-v1_5/train_llava_dpo_post.py \
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

python post_interaction_block/models/llava-v1_5/replace_bin.py --tune_stage 1 \
    --path_model_state_dict ${target_folder1}/pytorch_model-00002-of-00002.bin \
    --path_non_lora_state_dict ${output_dir1}/non_lora_trainables.bin

wait

CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu1} --master_port 12546 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
    --coco_path post_interaction_block/data/coco2014 \
    --pope_path post_interaction_block/data/POPE \
    --model-path ${target_folder1} \
    --set popular


wait

CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu1} --master_port 12546 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
    --coco_path post_interaction_block/data/coco2014 \
    --pope_path post_interaction_block/data/POPE \
    --model-path ${target_folder1} \
    --set random

wait

CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu1} --master_port 12546 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
    --coco_path post_interaction_block/data/coco2014 \
    --pope_path post_interaction_block/data/POPE \
    --model-path ${target_folder1} \
    --set adv

wait

eval_folder1="${output_dir1}/evaluation"

# 检查 evaluation 文件夹是否已经存在
if [ ! -d "${eval_folder1}" ]; then
    # 如果 evaluation 文件夹不存在，则创建它
    mkdir -p "${eval_folder1}"
    echo "evaluation folder: ${eval_folder1} has been created"
fi

mv "${pope_file1}" "${eval_folder1}/pope_adv.jsonl"
mv "${pope_file2}" "${eval_folder1}/pope_random.jsonl"
mv "${pope_file3}" "${eval_folder1}/pope_popular.jsonl"

wait


# version2='v8'
# time2='20240517'
# train_bs2='2'
# eval_bs2='1'
# gradient_accumulation_steps2='16'
# epoch2='1'
# gpu2='4'
# lr2='2e-6'

# pope_file1="post_interaction_block/models/llava-v1_5/pope_adv.jsonl"
# pope_file2="post_interaction_block/models/llava-v1_5/pope_random.jsonl"
# pope_file3="post_interaction_block/models/llava-v1_5/pope_popular.jsonl"

# source_folder="/home/cuiruochen/model/llava-v1.5-7b-----------pib-to-be-copied"
# target_folder2="/home/cuiruochen/model/llava-v1.5-7b-post-${time2}-${version2}-bs-${train_bs2}-${eval_bs2}-${gradient_accumulation_steps2}-epoch-${epoch2}-gpu-${gpu2}-lr-${lr2}-gelu"
# output_dir2="post_interaction_block/models/llava-v1_5/checkpoints/llava-post-${time2}-${version2}-bs-${train_bs2}-${eval_bs2}-${gradient_accumulation_steps2}-epoch-${epoch2}-gpu-${gpu2}-lr-${lr2}-gelu"
# # 检查目标文件夹路径是否正确
# echo "target_folder：$target_folder2"

# # 复制文件夹到目标路径, 如果已经存在路径, 会嵌套复制. 所以需要检查是否已经存在
# if [ -d "$target_folder2" ]; then
#     echo "Target folder already exists. Skipping copy."
# else
#     # 目标路径不存在，执行复制
#     cp -r "$source_folder" "$target_folder2"

#     # 检查复制是否成功
#     if [ $? -eq 0 ]; then
#         echo "Folder copied successfully!"
#     else
#         echo "Folder copy failed!"
#         exit 1
#     fi
# fi
# echo "Folders check completed"

# deepspeed --include localhost:${localhost} --master_port 12141 post_interaction_block/models/llava-v1_5/train_llava_dpo_post.py \
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
#     --tune_post_interaction_block True \
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

# wait

# python post_interaction_block/models/llava-v1_5/replace_bin.py --tune_stage 1 \
#     --path_model_state_dict ${target_folder2}/pytorch_model-00002-of-00002.bin \
#     --path_non_lora_state_dict ${output_dir2}/non_lora_trainables.bin

# wait

# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu2} --master_port 12538 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
#     --coco_path post_interaction_block/data/coco2014 \
#     --pope_path post_interaction_block/data/POPE \
#     --model-path ${target_folder2} \
#     --set popular


# wait

# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu2} --master_port 12538 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
#     --coco_path post_interaction_block/data/coco2014 \
#     --pope_path post_interaction_block/data/POPE \
#     --model-path ${target_folder2} \
#     --set random

# wait

# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu2} --master_port 12538 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
#     --coco_path post_interaction_block/data/coco2014 \
#     --pope_path post_interaction_block/data/POPE \
#     --model-path ${target_folder2} \
#     --set adv

# wait

# eval_folder2="${output_dir2}/evaluation"

# # 检查 evaluation 文件夹是否已经存在
# if [ ! -d "${eval_folder2}" ]; then
#     # 如果 evaluation 文件夹不存在，则创建它
#     mkdir -p "${eval_folder2}"
#     echo "evaluation folder: ${eval_folder2} has been created"
# fi

# mv "${pope_file1}" "${eval_folder2}/pope_adv.jsonl"
# mv "${pope_file2}" "${eval_folder2}/pope_random.jsonl"
# mv "${pope_file3}" "${eval_folder2}/pope_popular.jsonl"

# wait





# version3='v8'
# time3='20240517'
# train_bs3='2'
# eval_bs3='1'
# gradient_accumulation_steps3='16'
# epoch3='1'
# gpu3='4'
# lr3='2e-7'

# pope_file1="post_interaction_block/models/llava-v1_5/pope_adv.jsonl"
# pope_file2="post_interaction_block/models/llava-v1_5/pope_random.jsonl"
# pope_file3="post_interaction_block/models/llava-v1_5/pope_popular.jsonl"

# source_folder="/home/cuiruochen/model/llava-v1.5-7b-----------pib-to-be-copied"
# target_folder3="/home/cuiruochen/model/llava-v1.5-7b-post-${time3}-${version3}-bs-${train_bs3}-${eval_bs3}-${gradient_accumulation_steps3}-epoch-${epoch3}-gpu-${gpu3}-lr-${lr3}-gelu"
# output_dir3="post_interaction_block/models/llava-v1_5/checkpoints/llava-post-${time3}-${version3}-bs-${train_bs3}-${eval_bs3}-${gradient_accumulation_steps3}-epoch-${epoch3}-gpu-${gpu3}-lr-${lr3}-gelu"
# # 检查目标文件夹路径是否正确
# echo "target_folder: $target_folder3"

# # 复制文件夹到目标路径, 如果已经存在路径, 会嵌套复制. 所以需要检查是否已经存在
# if [ -d "$target_folder3" ]; then
#     echo "Target folder already exists. Skipping copy."
# else
#     # 目标路径不存在，执行复制
#     cp -r "$source_folder" "$target_folder3"

#     # 检查复制是否成功
#     if [ $? -eq 0 ]; then
#         echo "Folder copied successfully!"
#     else
#         echo "Folder copy failed!"
#         exit 1
#     fi
# fi
# echo "Folders check completed"

# deepspeed --include localhost:${localhost} --master_port 12157 post_interaction_block/models/llava-v1_5/train_llava_dpo_post.py \
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
#     --tune_post_interaction_block True \
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

# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu3} --master_port 12538 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
#     --coco_path post_interaction_block/data/coco2014 \
#     --pope_path post_interaction_block/data/POPE \
#     --model-path ${target_folder3} \
#     --set popular


# wait

# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu3} --master_port 12538 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
#     --coco_path post_interaction_block/data/coco2014 \
#     --pope_path post_interaction_block/data/POPE \
#     --model-path ${target_folder3} \
#     --set random

# wait

# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu3} --master_port 12538 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
#     --coco_path post_interaction_block/data/coco2014 \
#     --pope_path post_interaction_block/data/POPE \
#     --model-path ${target_folder3} \
#     --set adv

# wait

# eval_folder3="${output_dir3}/evaluation"

# # 检查 evaluation 文件夹是否已经存在
# if [ ! -d "${eval_folder3}" ]; then
#     # 如果 evaluation 文件夹不存在，则创建它
#     mkdir -p "${eval_folder3}"
#     echo "evaluation folder: ${eval_folder3} has been created"
# fi

# mv "${pope_file1}" "${eval_folder3}/pope_adv.jsonl"
# mv "${pope_file2}" "${eval_folder3}/pope_random.jsonl"
# mv "${pope_file3}" "${eval_folder3}/pope_popular.jsonl"

# wait



# version4='v8'
# time4='20240508'
# train_bs4='2'
# eval_bs4='1'
# gradient_accumulation_steps4='16'
# epoch4='1'
# gpu4='4'
# lr4='5e-7'

# pope_file1="post_interaction_block/models/llava-v1_5/pope_adv.jsonl"
# pope_file2="post_interaction_block/models/llava-v1_5/pope_random.jsonl"
# pope_file3="post_interaction_block/models/llava-v1_5/pope_popular.jsonl"

# source_folder="/home/cuiruochen/model/llava-v1.5-7b-----------pib-to-be-copied"
# target_folder4="/home/cuiruochen/model/llava-v1.5-7b-post-${time4}-${version4}-bs-${train_bs4}-${eval_bs4}-${gradient_accumulation_steps4}-epoch-${epoch4}-gpu-${gpu4}-lr-${lr4}"
# output_dir4="post_interaction_block/models/llava-v1_5/checkpoints/llava-post-${time4}-${version4}-bs-${train_bs4}-${eval_bs4}-${gradient_accumulation_steps4}-epoch-${epoch4}-gpu-${gpu4}-lr-${lr4}"
# # 检查目标文件夹路径是否正确
# echo "target_folder：$target_folder4"

# # 复制文件夹到目标路径, 如果已经存在路径, 会嵌套复制. 所以需要检查是否已经存在
# if [ -d "$target_folder4" ]; then
#     echo "Target folder already exists. Skipping copy."
# else
#     # 目标路径不存在，执行复制
#     cp -r "$source_folder" "$target_folder4"

#     # 检查复制是否成功
#     if [ $? -eq 0 ]; then
#         echo "Folder copied successfully!"
#     else
#         echo "Folder copy failed!"
#         exit 1
#     fi
# fi
# echo "Folders check completed"

# deepspeed --include localhost:${localhost} --master_port 12149 post_interaction_block/models/llava-v1_5/train_llava_dpo_post.py \
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
#     --tune_post_interaction_block True \
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

# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu4} --master_port 12546 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
#     --coco_path post_interaction_block/data/coco2014 \
#     --pope_path post_interaction_block/data/POPE \
#     --model-path ${target_folder4} \
#     --set popular


# wait

# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu4} --master_port 12546 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
#     --coco_path post_interaction_block/data/coco2014 \
#     --pope_path post_interaction_block/data/POPE \
#     --model-path ${target_folder4} \
#     --set random

# wait

# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu4} --master_port 12546 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
#     --coco_path post_interaction_block/data/coco2014 \
#     --pope_path post_interaction_block/data/POPE \
#     --model-path ${target_folder4} \
#     --set adv

# wait

# eval_folder4="${output_dir4}/evaluation"

# # 检查 evaluation 文件夹是否已经存在
# if [ ! -d "${eval_folder4}" ]; then
#     # 如果 evaluation 文件夹不存在，则创建它
#     mkdir -p "${eval_folder4}"
#     echo "evaluation folder: ${eval_folder4} has been created"
# fi

# mv "${pope_file1}" "${eval_folder4}/pope_adv.jsonl"
# mv "${pope_file2}" "${eval_folder4}/pope_random.jsonl"
# mv "${pope_file3}" "${eval_folder4}/pope_popular.jsonl"


wait

# export CUDA_VISIBLE_DEVICES=${localhost}

# python 哥们,留四张卡,跑大模型.py --size 30000 --gpus ${gpu2} --interval 0.01

CUDA_VISIBLE_DEVICES=${localhost} python 哥们,留四张卡,跑大模型.py --size 30000 --gpus ${gpu1} --interval 0.01

