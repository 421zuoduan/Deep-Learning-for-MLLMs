import torch

# # 加载 non_lora_trainables.bin 文件中的 lm_head 参数
non_lora_state_dict = torch.load("/home/cuiruochen/HA-DPO/ha_dpo/results/train_postv3-20240406-bs-1-1-16-epoch-3/llava-post-decoder-bs-1-1-16-epoch-3-gpu-4/non_lora_trainables.bin")

print(non_lora_state_dict.keys())

# llm_state_dict = torch.load("/home/cuiruochen/model/llava-v1.5-7b-train_postv1-20240331-bs-1-1-16/pytorch_model-00002-of-00002.bin")

# print(llm_state_dict.keys())

# print(llm_state_dict['post_decoder.align.fc1.weight'])
