CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master-port 17451 post_interaction_block/models/minigpt4/pope_eval.py \
    --cfg-path post_interaction_block/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml \
    --llama-model /home/cuiruochen/model/minigpt4/llama-2-7b-chat-hf \
    --set popular \
    --pope-path post_interaction_block/data/POPE \
    --coco-path post_interaction_block/data/coco2014

wait

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master-port 17451 post_interaction_block/models/minigpt4/pope_eval.py \
    --cfg-path post_interaction_block/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml \
    --llama-model /home/cuiruochen/model/minigpt4/llama-2-7b-chat-hf \
    --set random \
    --pope-path post_interaction_block/data/POPE \
    --coco-path post_interaction_block/data/coco2014

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master-port 17451 post_interaction_block/models/minigpt4/pope_eval.py \
    --cfg-path post_interaction_block/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml \
    --llama-model /home/cuiruochen/model/minigpt4/llama-2-7b-chat-hf \
    --set adv \
    --pope-path post_interaction_block/data/POPE \
    --coco-path post_interaction_block/data/coco2014

CUDA_VISIBLE_DEVICES=0,1,2,3 python 哥们,留四张卡,跑大模型.py --size 30000 --gpus 4 --interval 0.01