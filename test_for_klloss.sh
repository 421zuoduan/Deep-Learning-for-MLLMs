devices=0,1,2,3

CUDA_VISIBLE_DEVICES=$devices torchrun --nproc_per_node 4 --master_port 15631 post_interaction_block/models/llava-v1_5/pope_eval_post.py \
    --coco_path post_interaction_block/data/coco2014 \
    --pope_path post_interaction_block/data/POPE \
    --model-path /home/cuiruochen/model/llava-v1.5-7b-post-20240507-v8-bs-2-1-16-epoch-1-gpu-4-lr-5e-7 \
    --set popular

python 哥们,留四张卡,跑大模型.py --size 30000 --gpus 4 --interval 0.01


torchrun --nproc_per_node 4 --master_port 15631 post_interaction_block/models/llava-v1_5/pope_eval.py \
    --coco_path post_interaction_block/data/coco2014 \
    --pope_path post_interaction_block/data/POPE \
    --model-path /home/cuiruochen/model/llava-v1.5-7b \
    --set popular

