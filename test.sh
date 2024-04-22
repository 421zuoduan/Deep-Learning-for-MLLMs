
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port $RANDOM ha_dpo/models/llava-v1_5/pope_eval_post.py \
    --coco_path ha_dpo/data/coco2014 \
    --pope_path ha_dpo/data/POPE \
    --model-path /home/cuiruochen/model/llava-v1.5-7b-train_postv8-20240422-bs-2-1-16-epoch-2-gpu-4-lr-2e-6 \
    --set popular

python 哥们,留四张卡,跑大模型.py --size 30000 --gpus 4 --interval 0.01