# 假设我们有一个中文文本字符串
import torch
from torch import nn

text = "你好世界，这是一个简单的例子。"

# 将每个中文字符映射到一个唯一的索引
chars = sorted(list(set(text)))
char_to_index = {ch: idx for idx, ch in enumerate(chars)}
index_to_char = {idx: ch for ch, idx in char_to_index.items()}

# 将文本转换为整数序列
maxlen = 5  # 每个输入序列的长度
step = 1    # 步长
encoded = [char_to_index[ch] for ch in text]

sequences = []
next_chars = []
for i in range(0, len(encoded) - maxlen, step):
    sequences.append(encoded[i : i + maxlen])
    next_chars.append(encoded[i + maxlen])

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

model = LSTM(1,1024,1,1,1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 初始化最小损失为一个很大的数
min_loss = float('inf')
# 设置保存模型的路径
model_path = './model.pth'

for epoch in range(300):
    for i in range(len(sequences)):  # len(sequences)
        X_train = [sequences[i]]
        Y_train = [next_chars[i]]
        X_train = torch.tensor(X_train).float().unsqueeze(2)
        Y_train = torch.tensor(Y_train).float()
        # 累计损失
        total_loss = 0
        for j in range(len(X_train)):
            outputs = model(X_train[j].unsqueeze(0))  # 增加一个批次维度
            loss = criterion(outputs, Y_train[j].unsqueeze(0))
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累计损失
            total_loss += loss.item()
    # 计算平均损失
    avg_loss = total_loss / len(next_chars)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

    # 如果当前epoch的损失小于之前的最小损失，则保存模型
    if avg_loss < min_loss:
        min_loss = avg_loss
        # 保存模型
        torch.save(model.state_dict(), model_path)
        print(f'Model saved at epoch {epoch + 1} with loss {avg_loss}')


def wenben(model, text, index):
    if index == 10 :  return
    encoded1 = [char_to_index[ch] for ch in text]
    test_input1 = torch.tensor([encoded1]).float().unsqueeze(2)
    outputs1 = model(test_input1)
    char_id = round(outputs1.item())
    char_cur = index_to_char[char_id]
    text_new = text + char_cur
    print(f"预测字符: {char_cur}")
    index+=1
    wenben(model, text_new[1:],index)


model.load_state_dict(torch.load("./model.pth"))
model.eval()
text = "你好世界，"
index = 0
wenben(model,text,index)


