import torch
state_dict = torch.load("/home/cuiruochen/model/InstructBLIP/eva_vit_g/eva_vit_g.pth")
print(type(state_dict))
 
for i in state_dict:
    print(i)
    print(type(state_dict[i]))
    print("aa:",state_dict[i].data.size())
    print("bb:",state_dict[i].requires_grad)
    
    # 如果block.2在state_dict[i]中，就停止
    if "blocks.2" in i:
        break