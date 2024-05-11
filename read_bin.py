import torch

# # 加载 non_lora_trainables.bin 文件中的 lm_head 参数
# non_lora_state_dict = torch.load("/home/cuiruochen/HA-DPO/ha_dpo/models/llava-v1_5/checkpoints/llava-post-decoder-20240409-v5-bs-1-1-16-epoch-1-gpu-4-stages-2/stage-2/non_lora_trainables.bin")

# print(non_lora_state_dict.keys())

# llm_state_dict = torch.load("/home/cuiruochen/model/llava-v1.5-7b-train_postv4-20240407-bs-1-1-16-epoch-1-gpu-4-stages-2/pytorch_model-00002-of-00002.bin")

llm_state_dict = torch.load("/home/cuiruochen/model/InstructBLIP/bert-base-uncased/pytorch_model.bin")

print(llm_state_dict.keys())

# print(llm_state_dict['post_decoder.align.fc1.weight'])

# print(f"model.lm_head.weight: {llm_state_dict['lm_head.weight']}")