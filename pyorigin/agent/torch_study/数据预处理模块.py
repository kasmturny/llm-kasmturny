import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
'''
就是循环遍历多少轮epoch_num
然后每次循环epoch遍历多少个batch,知道当前epoch的batch遍历完
'''
# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, X_data, Y_data):
        """
        初始化数据集，X_data 和 Y_data 是两个列表或数组
        X_data: 输入特征
        Y_data: 目标标签
        """
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        """返回数据集的大小"""
        return len(self.X_data)

    def __getitem__(self, idx):
        """返回指定索引的数据"""
        x = torch.tensor(self.X_data[idx], dtype=torch.float32)  # 转换为 Tensor
        y = torch.tensor(self.Y_data[idx], dtype=torch.float32)
        return x, y

# 示例数据
X_data = [[1, 2], [3, 4], [5, 6], [7, 8]]  # 输入特征
Y_data = [1, 0, 1, 0]  # 目标标签
# 创建数据集实例
dataset = MyDataset(X_data, Y_data)
print('')
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
epoch_num = 5
for epoch in range(epoch_num):
    for inputs, targets in dataloader:
        print(inputs)
        print(targets)
        # 这里是模型训练的代码
        # inputs 是当前批量的输入特征
        # targets 是当前批量的目标标签

