CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 post_interaction_block/models/instructblip/pope_eval_post.py \
    --cfg-path post_interaction_block/models/instructblip/vigc/projects/post_interaction_block_hadpo/instruct_vicuna7b_test.yaml \
    --llm-model /home/cuiruochen/model/InstructBLIP/vicuna-7b-v1.1 \
    --set popular \
    --pope-path post_interaction_block/data/POPE \
    --coco-path post_interaction_block/data/coco2014

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 post_interaction_block/models/instructblip/pope_eval_post.py \
    --cfg-path post_interaction_block/models/instructblip/vigc/projects/post_interaction_block_hadpo/instruct_vicuna7b_test.yaml \
    --llm-model /home/cuiruochen/model/InstructBLIP/vicuna-7b-v1.1 \
    --set random \
    --pope-path post_interaction_block/data/POPE \
    --coco-path post_interaction_block/data/coco2014

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 post_interaction_block/models/instructblip/pope_eval_post.py \
    --cfg-path post_interaction_block/models/instructblip/vigc/projects/post_interaction_block_hadpo/instruct_vicuna7b_test.yaml \
    --llm-model /home/cuiruochen/model/InstructBLIP/vicuna-7b-v1.1 \
    --set adv \
    --pope-path post_interaction_block/data/POPE \
    --coco-path post_interaction_block/data/coco2014

CUDA_VISIBLE_DEVICES=0,1,2,3 python 哥们,留四张卡,跑大模型.py --size 30000 --gpus 4 --interval 0.01