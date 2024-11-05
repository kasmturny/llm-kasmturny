"""
运行这串代码之后，在c/user/.cache/huggingface/hub中会生成bert-base-chinese文件夹
"""

from transformers import BertTokenizer, BertModel


# 加载预训练的分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 加载预训练的BERT模型
model = BertModel.from_pretrained('bert-base-chinese')

# 定义文本
text = "人工智能是计算机科学的一个分支。"

# 使用分词器对文本进行编码
encoded_input = tokenizer(text, return_tensors='pt')

# 使用模型处理编码后的文本
outputs = model(**encoded_input)

# 获取模型的输出
last_hidden_state = outputs.last_hidden_state
pooler_output = outputs.pooler_output

# 打印输出
print("Last Hidden State:", last_hidden_state)
print("Pooler Output:", pooler_output)