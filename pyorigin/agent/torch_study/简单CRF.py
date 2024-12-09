import torch
import torch.nn as nn
from torchcrf import CRF

# 假设我们有以下标签和词汇表
tags = ['B', 'I', 'O']
vocab = ['我', '爱', '北京', '天安门']

# 将词汇和标签映射为索引
word_to_ix = {word: i for i, word in enumerate(vocab)}
tag_to_ix = {tag: i for i, tag in enumerate(tags)}

# 假设我们有一个句子
sentence = ['我', '爱', '北京', '天安门']

# 将句子转换为索引
sentence_in = torch.tensor([[word_to_ix[w] for w in sentence]])

# 假设我们有一个随机的标注序列
gold_tags = torch.tensor([[tag_to_ix['B'], tag_to_ix['I'], tag_to_ix['O'], tag_to_ix['O']]])

# 定义一个简单的CRF模型
class SimpleCRF(nn.Module):
    def __init__(self, tagset_size):
        super(SimpleCRF, self).__init__()
        self.crf = CRF(tagset_size)

    def forward(self, emissions, tags):
        loss = -self.crf(emissions, tags)
        return loss

    def predict(self, emissions):
        tags = self.crf.decode(emissions)
        return tags

# 初始化模型
model = SimpleCRF(len(tag_to_ix))

# 随机初始化emissions
emissions = torch.randn(1, len(sentence), len(tag_to_ix))

emissions = torch.tensor([
  [
    [0.7, 0.2, 0.1],  # 对于序列中的第一个元素，标签B的概率最高
    [0.7, 0.2, 0.1],  # 对于序列中的第二个元素，标签B的概率最高
    [0.1, 0.2, 0.7],  # 对于序列中的第三个元素，标签O的概率最高
    [0.1, 0.2, 0.7]   # 对于序列中的第四个元素，标签O的概率最高
  ]
])
"""
在自然语言处理（NLP）和序列标注任务中，发射矩阵（emissions）的形状通常与输入序列的长度、标签的数量以及批次的大小有关。对于一个形状为1, 4, 3的发射矩阵，这三个维度可以解释如下：

批次大小（Batch Size）: 第一个维度的大小是1，这意味着当前处理的批次中只有一个序列。在NLP任务中，批次大小通常表示同时处理的数据点的数量。在这个例子中，我们只处理一个序列。
序列长度（Sequence Length）: 第二个维度的大小是4，这意味着当前处理的序列由4个元素组成。在文本处理中，这通常意味着序列中有4个词或字符。
标签数量（Number of Tags）: 第三个维度的大小是3，这意味着存在3种不同的标签。在BIO标注任务中，这通常对应于B（开始）、I（内部）和O（外部）标签，或者是其他具体的标签集合。
因此，一个形状为1, 4, 3的发射矩阵可以表示为：

[
  [
    [0.1, 0.2, 0.7],  # 对于序列中的第一个元素，标签B、I、O的概率
    [0.4, 0.1, 0.5],  # 对于序列中的第二个元素，标签B、I、O的概率
    [0.3, 0.6, 0.1],  # 对于序列中的第三个元素，标签B、I、O的概率
    [0.2, 0.8, 0.0]   # 对于序列中的第四个元素，标签B、I、O的概率
  ]
]
在这个矩阵中，每一行对应于序列中的一个元素，每一列对应于一个标签的概率。
"""



# 训练模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    optimizer.zero_grad()
    loss = model(emissions, gold_tags)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 使用模型进行预测
with torch.no_grad():
    pred_tags = model.predict(emissions)
    print('Predicted tags:', [tags[tag[0]] for tag in pred_tags])
