#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 15691 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
    --coco_path post_interaction_block/data/coco2014 \
    --pope_path post_interaction_block/data/POPE \
    --model-path /home/cuiruochen/model/llava-v1.5-7b-post-20240511-v8-bs-2-1-16-epoch-1-gpu-4-lr-5e-7-align-after-align \
    --set popular

# python 哥们,留四张卡,跑大模型.py --size 30000 --gpus 4 --interval 0.01

CUDA_VISIBLE_DEVICES=4,5,6,7 python 哥们,留四张卡,跑大模型.py --size 30000 --gpus 4 --interval 0.01

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 42142 ha_dpo/models/llava-v1_5/generate_ans.py \
#     --coco_path ha_dpo/data/coco2014 \
#     --pope_path ha_dpo/data/POPE \
#     --model-path ha_dpo/models/llava-v1_5/checkpoints/llava-post-decoder-20240501-v8-bs-4-1-8-epoch-1-gpu-4-lr-2e-6 \
#     --set popular

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 42142 ha_dpo/models/llava-v1_5/pope_eval_post.py \
#     --coco_path ha_dpo/data/coco2014 \
#     --pope_path ha_dpo/data/POPE \
#     --model-path ha_dpo/models/llava-v1_5/checkpoints/llava-post-decoder-20240501-v8-bs-4-1-8-epoch-1-gpu-4-lr-2e-6 \
#     --set popular