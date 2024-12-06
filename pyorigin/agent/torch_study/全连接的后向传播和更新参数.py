import torch
import torch.nn as nn
"""
定义模型model
前向传播out = model.forward(x)
计算loss(out, y)
后向传播loss.backward()
更新参数optimizer.step()
"""

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 2)  # 输入层到隐藏层
        self.fc2 = nn.Linear(2, 2)  # 隐藏层到输出层
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数
        self.custom_weights()  # 自定义权重和偏置

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
        print("\033[0;31;40m每一层输出\033[0m")
        x = self.fc1(x)
        x = self.sigmoid(x)  # 应用Sigmoid激活函数
        print(x)  # tensor([0.7858, 0.8176], grad_fn=<SigmoidBackward0>)
        x = self.fc2(x)
        x = self.sigmoid(x)  # 通常在输出层使用Sigmoid
        print(x)  # tensor([0.6982, 0.6694], grad_fn=<SigmoidBackward0>)
        return x

    def train_fn(self, x, y):
        loss_fn = nn.MSELoss()  # 均方误差损失函数
        optimizer = torch.optim.SGD(self.parameters(), lr=1)  # 随机梯度下降优化器
        print("\033[0;31;40m权重\033[0m")
        print("fc1 weights:\n", model.fc1.weight)
        print("fc1 biases:\n", model.fc1.bias)
        print("fc2 weights:\n", model.fc2.weight)
        print("fc2 biases:\n", model.fc2.bias)
        output = self.forward(x)
        loss = loss_fn(output, y)
        print(f"\033[0;31;40m正向传播得到Loss:{loss.item()}\033[0m")
        loss.backward()  # 这个看来仅仅只是梯度计算，并不更新梯度
        optimizer.step()  # 这个函数才是更新梯度的，只有这步之后才有训练效果
        print("\033[0;31;40m权重更新完毕\n\033[0m")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
x = torch.tensor([3.0, 0.0, 1.0], dtype=torch.float32).to(device)
y = torch.tensor([1.0, 0.0], dtype=torch.float32).to(device)

for i in range(62):
    model.train_fn(x, y)
# 大概62轮之后，loss变为0，模型参数就固定了，这个时候模型就训练好了，当然只是针对输入【3，0，1】^T，输出【1，0】^T的情况
