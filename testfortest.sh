version1='v8'
time1='20240507'
train_bs1='2'
eval_bs1='1'
gradient_accumulation_steps1='16'
epoch1='1'
localhost='0,1,2,3'
gpu1='4'
lr1='5e-7'


target_folder1="/home/cuiruochen/model/llava-v1.5-7b-post-${time1}-${version1}-bs-${train_bs1}-${eval_bs1}-${gradient_accumulation_steps1}-epoch-${epoch1}-gpu-${gpu1}-lr-${lr1}"
output_dir1="post_interaction_block/models/llava-v1_5/checkpoints/llava-post-${time1}-${version1}-bs-${train_bs1}-${eval_bs1}-${gradient_accumulation_steps1}-epoch-${epoch1}-gpu-${gpu1}-lr-${lr1}-ablation-llava-post-sa"


CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${gpu1} --master_port 42142 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
    --coco_path post_interaction_block/data/coco2014 \
    --pope_path post_interaction_block/data/POPE \
    --model-path ${target_folder1} \
    --set popular


wait



CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port $RANDOM post_interaction_block/models/llava-v1_5/generate_ans.py     --coco_path post_interaction_block/data/coco2014     --pope_path post_interaction_block/data/POPE     --model-path /home/cuiruochen/model/llava-v1.5-7b     --set popular    --K 100



export CUDA_VISIBLE_DEVICES=${localhost}

python 哥们,留四张卡,跑大模型.py --size 30000 --gpus ${gpu1} --interval 0.01


