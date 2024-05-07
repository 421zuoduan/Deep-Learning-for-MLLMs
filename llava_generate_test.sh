localhost='0,1,2,3'
nproc_per_node=4
master_port=42142
K=1

CUDA_VISIBLE_DEVICES=${localhost} torchrun --nproc_per_node ${nproc_per_node} --master_port ${master_port} post_interaction_block/models/llava-v1_5/generate_ans.py \
    --coco_path post_interaction_block/data/coco2014 \
    --pope_path post_interaction_block/data/POPE \
    --model-path /home/cuiruochen/model/llava-v1.5-7b-train_postv8-20240423-bs-2-1-16-epoch-2-gpu-4-lr-5e-7 \
    --K ${K} \
    --set popular