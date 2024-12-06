import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 2)  # 输入层到隐藏层
        self.fc2 = nn.Linear(2, 2)  # 隐藏层到输出层
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

        # 假设你有一些具体的权重和偏置值
        self.custom_weights()

    def custom_weights(self):
        # 设置fc1层的权重和偏置
        # 假设weights_fc1是一个形状为(2, 3)的张量
        # 假设bias_fc1是一个形状为(2,)的张量
        weights_fc1 = [[0.1, 0.2, 0.6], [0.4, 0.3, 0.1]]
        bias_fc1 = [0.4, 0.2]
        self.fc1.weight.data = torch.tensor(weights_fc1, dtype=torch.float32)
        self.fc1.bias.data = torch.tensor(bias_fc1, dtype=torch.float32)

        # 设置fc2层的权重和偏置
        # 假设weights_fc2是一个形状为(2, 2)的张量
        # 假设bias_fc2是一个形状为(2,)的张量
        weights_fc2 = [[0.2, 0.1], [0.1, 0.4]]
        bias_fc2 = [0.6, 0.3]
        self.fc2.weight.data = torch.tensor(weights_fc2, dtype=torch.float32)
        self.fc2.bias.data = torch.tensor(bias_fc2, dtype=torch.float32)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)  # 应用Sigmoid激活函数
        print(x)  # tensor([0.7858, 0.8176], grad_fn=<SigmoidBackward0>)
        x = self.fc2(x)
        x = self.sigmoid(x)  # 通常在输出层使用Sigmoid
        print(x)  # tensor([0.6982, 0.6694], grad_fn=<SigmoidBackward0>)
        return x

# 正向传播，得到结果
model = SimpleNN()
input_data = torch.tensor([3.0, 0.0, 1.0], dtype=torch.float32)
output = model.forward(input_data)
