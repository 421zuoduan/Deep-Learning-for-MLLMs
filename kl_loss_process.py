# 统计 KL 散度的分布情况
# 从 kl_divergence.txt 中读取数据，统计 KL 散度的分布情况, 如果小于0则置为0
# 按照 0.0001 为间隔, 统计每个间隔中的数据量
# 可视化为直方图
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
with open('kl_divergence.txt', 'r') as f:
    kl = f.readlines()
kl = [float(i) for i in kl]
kl = np.array(kl)
# 将 kl 中小于 0 的都置为 0
kl[kl <= 0] = 0
print(kl.shape)
# kl = kl[kl >= 0]
# kl = kl[kl <= 0.1] 
# 统计数据
intersection = np.arange(0, 0.019, 0.0001)
count = []
for i in range(len(intersection)-1):
    count.append(np.sum((kl >= intersection[i]) & (kl < intersection[i+1])))

# 可视化
# plt.bar(intersection[:-1], count, width=0.0001)
# plt.xlabel('KL divergence')
# plt.ylabel('Count')
# plt.title('KL divergence distribution')
# # plt.show()
# # 保存图片
# plt.savefig('kl_divergence_distribution.png')

# print(count)

# 按照上述区间统计每个区间产生幻觉的样本数, 可是化为柱状图做对比
# 真实值
path_true='/home/cuiruochen/HA-DPO/post_interaction_block/data/POPE/output/coco/coco_pope_popular.json'
path_eval='post_interaction_block/models/llava-v1_5/checkpoints/llava-post-20240507-v8-bs-2-1-16-epoch-1-gpu-4-lr-5e-7/POPE_evaluation/pope_popular.jsonl'
# 读取上述两个文件, 两个文件都是字典格式, 第一个文件 key 为 label 的值与第二个文件 key 为 answer 的前两个字符对比, 一致则该样本正确, 否则产生幻觉
import json
with open(path_true, 'r') as f:
    true = [json.loads(i) for i in f.readlines()]
with open(path_eval, 'r') as f: 
    eval = [json.loads(i) for i in f.readlines()]


print(len(true),true[0]['label'][0],len(eval),eval[0]['answer'][0])
# 统计上面区间 intersection 每个区间的幻觉样本数, 测试集的首字母是小写, 真实值的首字母是大写, 注意转换
count_true = []
count_eval = []
for i in range(len(intersection)-1):
    count_true.append(np.sum((kl >= intersection[i]) & (kl < intersection[i+1]) & (np.array([i['label'][0] for i in true]) == np.array([i['answer'][0].lower() for i in eval]))))
    count_eval.append(np.sum((kl >= intersection[i]) & (kl < intersection[i+1]) & (np.array([i['label'][0] for i in true]) != np.array([i['answer'][0].lower() for i in eval]))))
            