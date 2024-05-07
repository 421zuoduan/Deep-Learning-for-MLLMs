#!/usr/bin/env python 
# -*- coding:utf-8 -*- 
'''
 * @Author: Ruochen Cui 
 * @Date: 2024-03-27 17:35:53 
 * @Last Modified by:   Ruochen Cui 
 * @Last Modified time: 2024-03-27 17:35:53 
 * @Desc: 
'''
# import torch

# # 加载 non_lora_trainables.bin 文件中的 lm_head 参数
# non_lora_state_dict = torch.load("/home/cuiruochen/HA-DPO/ha_dpo/results/train_postv3-20240406-bs-1-1-16-epoch-2-gpu-4/llava-post-decoder-bs-1-1-16-epoch-2-gpu-4/non_lora_trainables.bin")

# # 加载 pytorch_model-00002-of-00002.bin 文件中的参数
# model_state_dict = torch.load("/home/cuiruochen/model/llava-v1.5-7b-train_postv3-20240406-bs-1-1-16-epoch-2-gpu-4/pytorch_model-00002-of-00002.bin")

# # print(non_lora_state_dict.keys())
# # print(model_state_dict.keys())

# # 检查并添加非重复参数
# for key, value in non_lora_state_dict.items():
#     if key not in model_state_dict:
#         model_state_dict[key] = value
        
# # 替换 lm_head 参数
# if "lm_head.weight" in non_lora_state_dict and "lm_head.weight" in model_state_dict:
#     model_state_dict["lm_head.weight"] = non_lora_state_dict["lm_head.weight"]

# # 保存修改后的模型参数
# torch.save(model_state_dict, "/home/cuiruochen/model/llava-v1.5-7b-train_postv3-20240406-bs-1-1-16-epoch-2-gpu-4/pytorch_model-00002-of-00002.bin")

import argparse
import torch

def replace_bin(config):
    
    if config.tune_stage == 1:
        add_post_decoder = True
        replace_head = False
        replace_post_decoder = True
    elif config.tune_stage == 2:
        add_post_decoder = False
        replace_head = True
        replace_post_decoder = True
    
    # 加载 non_lora_trainables.bin 文件中的 lm_head 参数
    non_lora_state_dict = torch.load(config.path_non_lora_state_dict)

    # 加载 pytorch_model-00002-of-00002.bin 文件中的参数
    model_state_dict = torch.load(config.path_model_state_dict)

    if add_post_decoder:
        # 检查并添加非重复参数
        for key, value in non_lora_state_dict.items():
            if key not in model_state_dict:
                model_state_dict[key] = value
            
    if replace_head:
        # 替换 lm_head 参数
        if "lm_head.weight" in non_lora_state_dict and "lm_head.weight" in model_state_dict:
            model_state_dict["lm_head.weight"] = non_lora_state_dict["lm_head.weight"]
        else:
            raise ValueError("lm_head.weight not found in non_lora_state_dict")
        
    if replace_post_decoder:
        # 找到non_lora_state_dict中包括'post_decoder'字符的所有参数
        post_decoder_keys = [k for k in non_lora_state_dict.keys() if 'post_decoder'.lower() in k]
        
        # 对于找到的所有参数, 检查是否在model_state_dict中, 如果在则替换, 否则报错
        for key in post_decoder_keys:
            if key in model_state_dict and key in non_lora_state_dict:
                model_state_dict[key] = non_lora_state_dict[key]
            else:
                raise ValueError(f"{key} not found in model_state_dict")
        

    # 保存修改后的模型参数
    torch.save(model_state_dict, config.path_model_state_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_model_state_dict", type=str, default=None)
    parser.add_argument("--path_non_lora_state_dict", type=str, default=None)
    parser.add_argument("--tune_stage", type=int, default=1)
    args = parser.parse_args()
    
    replace_bin(args)