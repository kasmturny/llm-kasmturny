import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_directions = 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputseq):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)
        output, _ = self.lstm(inputseq, (h_0, c_0))
        pred = self.linear(output[:, -1, :])  # 只使用最后一个时间步的输出
        return pred

# 创建模型
model = LSTM(input_size=1, hidden_size=32, num_layers=3, output_size=1, batch_size=1)
"""
1、输入层：
    输入序列的每个时间步将有一个维度为 input_size=1 的向量。
2、第一个LSTM层：
    输入：每个时间步的1维向量。
    输出：每个时间步输出一个32维的隐藏状态向量（hidden_size=32）。
    这个32维向量将被传递到下一个LSTM层作为输入。
3、第二个LSTM层：
    输入：第一个LSTM层输出的32维隐藏状态向量。
    输出：再次输出一个32维的隐藏状态向量。
    这个32维向量继续传递到第三个LSTM层。
4、第三个LSTM层：
    输入：第二个LSTM层输出的32维隐藏状态向量。
    输出：输出一个32维的隐藏状态向量。
    在每个时间步，每个LSTM层都会输出一个32维的隐藏状态向量。但是，这些隐藏状态向量不会直接连接到输出层，而是作为序列处理的一部分在层与层之间传递。
5、输出层：
    输入：最后一个LSTM层的输出，即第三个LSTM层的32维隐藏状态向量。
    输出：通过一个全连接层（通常是线性层）将32维的隐藏状态向量映射到 output_size=1 的输出。
"""

# 定义训练数据
X_train = torch.tensor([[1, 2, 3, 4, 5]]).float().unsqueeze(2)
Y_train = torch.tensor([6]).float()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(1000):
    for i in range(len(X_train)):
        # 前向传播
        test = X_train[i].unsqueeze(0)
        outputs = model(X_train[i].unsqueeze(0))
        loss = criterion(outputs, Y_train[i].unsqueeze(0))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

model.eval()
test_input = torch.tensor([[1,2,3,4,5]]).float().unsqueeze(2)
print(model(test_input))


