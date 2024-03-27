import torch

# 加载 non_lora_trainables.bin 文件中的 lm_head 参数
non_lora_state_dict = torch.load("/home/cuiruochen/HA-DPO/ha_dpo/models/llava-v1_5/checkpoints/llava-origin/non_lora_trainables.bin")

# 加载 pytorch_model-00002-of-00002.bin 文件中的参数
model_state_dict = torch.load("/home/cuiruochen/model/llava-v1.5-7b-test-no-ref-model/pytorch_model-00002-of-00002.bin")

# 替换 lm_head 参数
if "lm_head.weight" in non_lora_state_dict and "lm_head.weight" in model_state_dict:
    model_state_dict["lm_head.weight"] = non_lora_state_dict["lm_head.weight"]

# 保存修改后的模型参数
torch.save(model_state_dict, "/home/cuiruochen/model/llava-v1.5-7b-test-no-ref-model/pytorch_model-00002-of-00002.bin")
