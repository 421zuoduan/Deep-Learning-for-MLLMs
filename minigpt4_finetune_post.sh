localhost='0,1,2,3'
# localhost='4,5,6,7'

version1='v8'
time1='20240510'
per_device_train_batch_size1='2'
eval_bs1='1'
gradient_accumulation_steps1='16'
epoch1='1'
gpu1='4'
learning_rate1='5e-7'

source_folder="/home/cuiruochen/model/minigpt4/vicuna-7b-v1.1--------pib-to-be-copied"
target_folder1="/home/cuiruochen/model/minigpt4/vicuna-7b-v1.1-post-${time1}-${version1}-bs-${per_device_train_batch_size1}-${eval_bs1}-${gradient_accumulation_steps1}-epoch-${epoch1}-gpu-${gpu1}-lr-${learning_rate1}"
output_dir1="post_interaction_block/models/minigpt4/minigpt4/output/vicuna-7b-v1.1-post-${time1}-${version1}-bs-${per_device_train_batch_size1}-${eval_bs1}-${gradient_accumulation_steps1}-epoch-${epoch1}-gpu-${gpu1}-lr-${learning_rate1}"
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

CUDA_VISIBLE_DEVICES=${localhost} accelerate launch --main_process_port 36672 post_interaction_block/models/minigpt4/train_dpo.py \
    --cfg_path post_interaction_block/models/minigpt4/train_configs/minigpt4_llama2_stage3_dpo_post.yaml \
    --auxilary True \
    --ccsbualign_data_path post_interaction_block/data/cc_sbu_align \
    --pope_train_data_path post_interaction_block/data/hadpo/minigpt4/pope_data.json \
    --desc_train_data_path post_interaction_block/data/hadpo/minigpt4/desc_data.json \
    --vg_path post_interaction_block/data/VG \
    --lora_r 64 \
    --gradient_checkpointing False \
    --per_device_train_batch_size ${per_device_train_batch_size1} \
    --learning_rate ${learning_rate1} \
    --beta 0.1 \
    --gamma 0.5 \
    --gradient_accumulation_steps ${gradient_accumulation_steps1} \
    --max_steps 1000 \
    --output_dir ${output_dir1} \
    --logging_steps 4

# deepspeed --include localhost:${localhost} --master_port 36652 post_interaction_block/models/instructblip/train_instructblip_dpo_post.py \
#     --cfg_path post_interaction_block/models/instructblip/vigc/projects/post_interaction_block_hadpo/instruct_vicuna7b.yaml \
#     --deepspeed post_interaction_block/models/instructblip/deep_scripts/zero3.json \
#     --pope_train_data_path post_interaction_block/data/hadpo/instructblip/pope_data.json \
#     --desc_train_data_path post_interaction_block/data/hadpo/instructblip/desc_data.json \
#     --vg_path post_interaction_block/data/VG \
#     --gradient_checkpointing False \
#     --num_train_epoch ${epoch1} \
#     --run_name "instructblip" \
#     --gradient_accumulation_steps ${gradient_accumulation_steps1} \
#     --learning_rate ${learning_rate1} \
#     --warmup_steps 0 \
#     --per_device_train_batch_size ${per_device_train_batch_size1} \
#     --output_dir ${target_folder1} \
#     --logging_steps 4

# python post_interaction_block/models/llava-v1_5/replace_bin.py --tune_stage 1 \
#     --path_model_state_dict ${target_folder1}/pytorch_model-00002-of-00002.bin \
#     --path_non_lora_state_dict ${output_dir1}/non_lora_trainables.bin

wait

# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu1} --master_port 12546 post_interaction_block/models/instructblip/pope_eval_post.py \
#     --cfg-path post_interaction_block/models/instructblip/vigc/projects/post_interaction_block_hadpo/instruct_vicuna7b.yaml \
#     --llm-model ${target_folder1} \
#     --set popular \
#     --pope-path post_interaction_block/data/POPE \
#     --coco-path post_interaction_block/data/coco2014

# wait

# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu1} --master_port 12546 post_interaction_block/models/instructblip/pope_eval_post.py \
#     --cfg-path post_interaction_block/models/instructblip/vigc/projects/post_interaction_block_hadpo/instruct_vicuna7b.yaml \
#     --llm-model ${target_folder1} \
#     --set random \
#     --pope-path post_interaction_block/data/POPE \
#     --coco-path post_interaction_block/data/coco2014

# wait

# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu1} --master_port 12546 post_interaction_block/models/instructblip/pope_eval_post.py \
#     --cfg-path post_interaction_block/models/instructblip/vigc/projects/post_interaction_block_hadpo/instruct_vicuna7b.yaml \
#     --llm-model ${target_folder1} \
#     --set adv \
#     --pope-path post_interaction_block/data/POPE \
#     --coco-path post_interaction_block/data/coco2014

wait

# CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu1} --master_port 12546 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
#     --coco_path post_interaction_block/data/coco2014 \
#     --pope_path post_interaction_block/data/POPE \
#     --model-path ${output_dir1} \
#     --set adv


wait

CUDA_VISIBLE_DEVICES=${localhost} python 哥们,留四张卡,跑大模型.py --size 30000 --gpus ${gpu1} --interval 0.01

