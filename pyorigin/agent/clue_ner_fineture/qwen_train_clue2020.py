import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
import numpy as np

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    label2id = {'O': 0, 'B-address': 1, 'B-book': 2, 'B-company': 3, 'B-game': 4, 'B-government': 5, 'B-movie': 6,
                'B-name': 7, 'B-organization': 8, 'B-position': 9, 'B-scene': 10, 'I-address': 11, 'I-book': 12,
                'I-company': 13, 'I-game': 14, 'I-government': 15, 'I-movie': 16, 'I-name': 17, 'I-organization': 18,
                'I-position': 19, 'I-scene': 20, 'S-address': 21, 'S-book': 22, 'S-company': 23, 'S-game': 24,
                'S-government': 25, 'S-movie': 26, 'S-name': 27, 'S-organization': 28, 'S-position': 29, 'S-scene': 30}
    messages = []
    data = np.load(origin_path, allow_pickle=True)
    input_texts = data["words"]
    output_labels = data["labels"]
    for i in range(len(input_texts)):
        input_text = input_texts[i]
        temp = ""
        for meta in input_text:
            temp += ''.join(meta)
        input_text = temp
        output_label = output_labels[i]
        id = []
        for meta in output_label:
            id.append(f'{label2id[meta]}')
        output = f"""{{"BIO": "{output_label}", "ID": "{id}"}}"""
        message = {
            "instruction": """你是一个文本实体识别领域的专家,你需要先理解labels的实体,理解id2label和label2id的对应关系,对input每一个字符进行打标,返回打标结果 ，以 json 格式输出,如 {"BIO": "['B-company', 'I-company', 'I-company', 'I-company', 'O', 'O', 'O', 'O', 'O', 'B-name', 'I-name', 'I-name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-name', 'I-name', 'I-name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']", "ID": "['3', '13', '13', '13', '0', '0', '0', '0', '0', '7', '17', '17', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '7', '17', '17', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']"}
            id2label = {0: 'O', 1: 'B-address', 2: 'B-book', 3: 'B-company', 4: 'B-game', 5: 'B-government', 6: 'B-movie', 7: 'B-name', 8: 'B-organization', 9: 'B-position', 10: 'B-scene', 11: 'I-address', 12: 'I-book', 13: 'I-company', 14: 'I-game', 15: 'I-government', 16: 'I-movie', 17: 'I-name', 18: 'I-organization', 19: 'I-position', 20: 'I-scene', 21: 'S-address', 22: 'S-book', 23: 'S-company', 24: 'S-game', 25: 'S-government', 26: 'S-movie', 27: 'S-name', 28: 'S-organization', 29: 'S-position', 30: 'S-scene'}
            label2id = {'O': 0, 'B-address': 1, 'B-book': 2, 'B-company': 3, 'B-game': 4, 'B-government': 5, 'B-movie': 6, 'B-name': 7, 'B-organization': 8, 'B-position': 9, 'B-scene': 10, 'I-address': 11, 'I-book': 12, 'I-company': 13, 'I-game': 14, 'I-government': 15, 'I-movie': 16, 'I-name': 17, 'I-organization': 18, 'I-position': 19, 'I-scene': 20, 'S-address': 21, 'S-book': 22, 'S-company': 23, 'S-game': 24, 'S-government': 25, 'S-movie': 26, 'S-name': 27, 'S-organization': 28, 'S-position': 29, 'S-scene': 30}
            labels = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene'] 
            注意: 1. 输出的每一行都必须是正确的 json 字符串.""",
            "input": input_text,
            "output": output,
        }
        messages.append(message)
    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
def process_func(example):
    """
    将数据集进行预处理, 处理成模型可以接受的格式
    """

    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    system_prompt = """你是一个文本实体识别领域的专家,你需要先理解labels的实体,理解id2label和label2id的对应关系,对input每一个字符进行打标,返回打标结果 ，以 json 格式输出,如 {"BIO": "['B-company', 'I-company', 'I-company', 'I-company', 'O', 'O', 'O', 'O', 'O', 'B-name', 'I-name', 'I-name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-name', 'I-name', 'I-name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']", "ID": "['3', '13', '13', '13', '0', '0', '0', '0', '0', '7', '17', '17', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '7', '17', '17', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']"}
            id2label = {0: 'O', 1: 'B-address', 2: 'B-book', 3: 'B-company', 4: 'B-game', 5: 'B-government', 6: 'B-movie', 7: 'B-name', 8: 'B-organization', 9: 'B-position', 10: 'B-scene', 11: 'I-address', 12: 'I-book', 13: 'I-company', 14: 'I-game', 15: 'I-government', 16: 'I-movie', 17: 'I-name', 18: 'I-organization', 19: 'I-position', 20: 'I-scene', 21: 'S-address', 22: 'S-book', 23: 'S-company', 24: 'S-game', 25: 'S-government', 26: 'S-movie', 27: 'S-name', 28: 'S-organization', 29: 'S-position', 30: 'S-scene'}
            label2id = {'O': 0, 'B-address': 1, 'B-book': 2, 'B-company': 3, 'B-game': 4, 'B-government': 5, 'B-movie': 6, 'B-name': 7, 'B-organization': 8, 'B-position': 9, 'B-scene': 10, 'I-address': 11, 'I-book': 12, 'I-company': 13, 'I-game': 14, 'I-government': 15, 'I-movie': 16, 'I-name': 17, 'I-organization': 18, 'I-position': 19, 'I-scene': 20, 'S-address': 21, 'S-book': 22, 'S-company': 23, 'S-game': 24, 'S-government': 25, 'S-movie': 26, 'S-name': 27, 'S-organization': 28, 'S-position': 29, 'S-scene': 30}
            labels = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene'] 
            注意: 1. 输出的每一行都必须是正确的 json 字符串."""
    instruction = tokenizer(
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
            instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)

    return response


model_id = "qwen/Qwen2-1.5B-Instruct"
model_dir = "./qwen/Qwen2-1___5B-Instruct"

# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download(model_id, cache_dir="./", revision="master")

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 加载、处理数据集和测试集
# train_dataset_path = "ccfbdci.jsonl"
# train_jsonl_new_path = "ccf_train.jsonl"
train_dataset_path = "./data/train.npz"
train_jsonl_new_path = "./data/train_new.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)

# 得到训练集
total_df = pd.read_json(train_jsonl_new_path, lines=True)
train_df = total_df[int(len(total_df) * 0.1):]
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="./output/Qwen2-NER",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

swanlab_callback = SwanLabCallback(
    project="Qwen2-NER-fintune",
    experiment_name="Qwen2-1.5B-Instruct-Clue2020",
    description="使用通义千问Qwen2-1.5B-Instruct模型在NER数据集clue2020上微调，实现关键实体识别任务,并返回标注结果。",
    config={
        "model": model_id,
        "model_dir": model_dir,
        "dataset": "clue2020",
    },
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()

# 用测试集的随机20条，测试模型
# 得到测试集
test_df = total_df[:int(len(total_df) * 0.1)].sample(n=20)

test_text_list = []
for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)
    messages.append({"role": "assistant", "content": f"{response}"})
    result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
    test_text_list.append(swanlab.Text(result_text, caption=response))

swanlab.log({"Prediction": test_text_list})
swanlab.finish()