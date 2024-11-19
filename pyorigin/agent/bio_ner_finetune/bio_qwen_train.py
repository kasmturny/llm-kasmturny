import ast
import json

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab


def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    data = np.load(origin_path, allow_pickle=True)
    data_len = len(data['words'])

    for i in tqdm(range(data_len)):
        word = data['words'][i]
        label = data['labels'][i]

        input_text = ''.join(word)
        input_list = []
        output_list = []
        for j in range(len(word)):
            input_list.append([word[j], j])
            output_list.append([word[j], j, label[j]])

        message = {
            "instruction": """Role:
    你是一个命名实体识别的专家，可以很好的识别文本中的实体信息，实体信息有十类：地址、书籍、公司、游戏、政府、电影、名称、组织、职位、场景
Parameters:
    input_list:{input_list}
    input_text:{input_text}
Workflow:
    1. 仔细分析input_text，结合语境，分析出文中的实体信息
    2. 根据input_list和input_text的对应关系，按照实体信息和标记规则对input_list进行标注
    3. 将标注结果设置为output_list,并按照输出格式进行进行输出，输出格式必须是json格式
Rule:
    1. 实体信息：['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
    2. 标注标签：["O", "B-address", "B-book", "B-company", 'B-game', 'B-government', 'B-movie', 'B-name', 'B-organization', 'B-position', 'B-scene', "I-address", "I-book", "I-company", 'I-game', 'I-government', 'I-movie', 'I-name', 'I-organization', 'I-position', 'I-scene', "S-address", "S-book", "S-company", 'S-game', 'S-government', 'S-movie', 'S-name', 'S-organization', 'S-position', 'S-scene']
    3. 标注规则：'O'用来标注非实体，如果实体长度大于1，则用'B-label'标记实体开始，用'I-label'标记实体中间和结尾，如果实体长度为1，则用'S-label'标记实体
Example:
    input:{{
            "input_text": "彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，"
            "input_list": "[['彭', 0], ['小', 1], ['军', 2], ['认', 3], ['为', 4], ['，', 5], ['国', 6], ['内', 7], ['银', 8], ['行', 9], ['现', 10], ['在', 11], ['走', 12], ['的', 13], ['是', 14], ['台', 15], ['湾', 16], ['的', 17], ['发', 18], ['卡', 19], ['模', 20], ['式', 21], ['，', 22], ['先', 23], ['通', 24], ['过', 25], ['跑', 26], ['马', 27], ['圈', 28], ['地', 29], ['再', 30], ['在', 31], ['圈', 32], ['的', 33], ['地', 34], ['里', 35], ['面', 36], ['选', 37], ['择', 38], ['客', 39], ['户', 40], ['，', 41]]"
            }}
    output:{{
            "output_list": "[['彭', 0, 'B-name'], ['小', 1, 'I-name'], ['军', 2, 'I-name'], ['认', 3, 'O'], ['为', 4, 'O'], ['，', 5, 'O'], ['国', 6, 'O'], ['内', 7, 'O'], ['银', 8, 'O'], ['行', 9, 'O'], ['现', 10, 'O'], ['在', 11, 'O'], ['走', 12, 'O'], ['的', 13, 'O'], ['是', 14, 'O'], ['台', 15, 'B-address'], ['湾', 16, 'I-address'], ['的', 17, 'O'], ['发', 18, 'O'], ['卡', 19, 'O'], ['模', 20, 'O'], ['式', 21, 'O'], ['，', 22, 'O'], ['先', 23, 'O'], ['通', 24, 'O'], ['过', 25, 'O'], ['跑', 26, 'O'], ['马', 27, 'O'], ['圈', 28, 'O'], ['地', 29, 'O'], ['再', 30, 'O'], ['在', 31, 'O'], ['圈', 32, 'O'], ['的', 33, 'O'], ['地', 34, 'O'], ['里', 35, 'O'], ['面', 36, 'O'], ['选', 37, 'O'], ['择', 38, 'O'], ['客', 39, 'O'], ['户', 40, 'O'], ['，', 41, 'O']]"
            }}
Output:
    {{
        "output_list":output_list
    }}""",
            "input": f"""{{"input_text":"{input_text}","input_list":"{input_list}"}}""",
            "output": f"""{{"output_list":"{output_list}"}}"""
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

    MAX_LENGTH = 384*5
    input_ids, attention_mask, labels = [], [], []
    system_prompt = """Role:
    你是一个命名实体识别的专家，可以很好的识别文本中的实体信息，实体信息有十类：地址、书籍、公司、游戏、政府、电影、名称、组织、职位、场景
Parameters:
    input_list:{input_list}
    input_text:{input_text}
Workflow:
    1. 仔细分析input_text，结合语境，分析出文中的实体信息
    2. 根据input_list和input_text的对应关系，按照实体信息和标记规则对input_list进行标注
    3. 将标注结果设置为output_list,并按照输出格式进行进行输出，输出格式必须是json格式
Rule:
    1. 实体信息：['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
    2. 标注标签：["O", "B-address", "B-book", "B-company", 'B-game', 'B-government', 'B-movie', 'B-name', 'B-organization', 'B-position', 'B-scene', "I-address", "I-book", "I-company", 'I-game', 'I-government', 'I-movie', 'I-name', 'I-organization', 'I-position', 'I-scene', "S-address", "S-book", "S-company", 'S-game', 'S-government', 'S-movie', 'S-name', 'S-organization', 'S-position', 'S-scene']
    3. 标注规则：'O'用来标注非实体，如果实体长度大于1，则用'B-label'标记实体开始，用'I-label'标记实体中间和结尾，如果实体长度为1，则用'S-label'标记实体
Example:
    input:{{
            "input_text": "彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，"
            "input_list": "[['彭', 0], ['小', 1], ['军', 2], ['认', 3], ['为', 4], ['，', 5], ['国', 6], ['内', 7], ['银', 8], ['行', 9], ['现', 10], ['在', 11], ['走', 12], ['的', 13], ['是', 14], ['台', 15], ['湾', 16], ['的', 17], ['发', 18], ['卡', 19], ['模', 20], ['式', 21], ['，', 22], ['先', 23], ['通', 24], ['过', 25], ['跑', 26], ['马', 27], ['圈', 28], ['地', 29], ['再', 30], ['在', 31], ['圈', 32], ['的', 33], ['地', 34], ['里', 35], ['面', 36], ['选', 37], ['择', 38], ['客', 39], ['户', 40], ['，', 41]]"
            }}
    output:{{
            "output_list": "[['彭', 0, 'B-name'], ['小', 1, 'I-name'], ['军', 2, 'I-name'], ['认', 3, 'O'], ['为', 4, 'O'], ['，', 5, 'O'], ['国', 6, 'O'], ['内', 7, 'O'], ['银', 8, 'O'], ['行', 9, 'O'], ['现', 10, 'O'], ['在', 11, 'O'], ['走', 12, 'O'], ['的', 13, 'O'], ['是', 14, 'O'], ['台', 15, 'B-address'], ['湾', 16, 'I-address'], ['的', 17, 'O'], ['发', 18, 'O'], ['卡', 19, 'O'], ['模', 20, 'O'], ['式', 21, 'O'], ['，', 22, 'O'], ['先', 23, 'O'], ['通', 24, 'O'], ['过', 25, 'O'], ['跑', 26, 'O'], ['马', 27, 'O'], ['圈', 28, 'O'], ['地', 29, 'O'], ['再', 30, 'O'], ['在', 31, 'O'], ['圈', 32, 'O'], ['的', 33, 'O'], ['地', 34, 'O'], ['里', 35, 'O'], ['面', 36, 'O'], ['选', 37, 'O'], ['择', 38, 'O'], ['客', 39, 'O'], ['户', 40, 'O'], ['，', 41, 'O']]"
            }}
Output:
    {{
        "output_list":output_list
    }}""",
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

def f1_score(y_true, y_pred, mode='test'):
    def end_of_chunk(prev_tag, tag, prev_type, type_):
        """Checks if a chunk ended between the previous and current word.

        Args:
            prev_tag: previous chunk tag.
            tag: current chunk tag.
            prev_type: previous type.
            type_: current type.

        Returns:
            chunk_end: boolean.
        """
        chunk_end = False

        if prev_tag == 'S':
            chunk_end = True
        # pred_label中可能出现这种情形
        if prev_tag == 'B' and tag == 'B':
            chunk_end = True
        if prev_tag == 'B' and tag == 'S':
            chunk_end = True
        if prev_tag == 'B' and tag == 'O':
            chunk_end = True
        if prev_tag == 'I' and tag == 'B':
            chunk_end = True
        if prev_tag == 'I' and tag == 'S':
            chunk_end = True
        if prev_tag == 'I' and tag == 'O':
            chunk_end = True

        if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
            chunk_end = True

        return chunk_end

    def start_of_chunk(prev_tag, tag, prev_type, type_):
        """Checks if a chunk started between the previous and current word.

        Args:
            prev_tag: previous chunk tag.
            tag: current chunk tag.
            prev_type: previous type.
            type_: current type.

        Returns:
            chunk_start: boolean.
        """
        chunk_start = False

        if tag == 'B':
            chunk_start = True
        if tag == 'S':
            chunk_start = True

        if prev_tag == 'S' and tag == 'I':
            chunk_start = True
        if prev_tag == 'O' and tag == 'I':
            chunk_start = True

        if tag != 'O' and tag != '.' and prev_type != type_:
            chunk_start = True

        return chunk_start
    def get_entities(seq):
        """
        Gets entities from sequence.

        Args:
            seq (list): sequence of labels.

        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).

        Example:
            seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
            get_entities(seq)
            [('PER', 0, 1), ('LOC', 3, 3)]
        """
        # for nested list
        if any(isinstance(s, list) for s in seq):
            seq = [item for sublist in seq for item in sublist + ['O']]
        prev_tag = 'O'
        prev_type = ''
        begin_offset = 0
        chunks = []
        for i, chunk in enumerate(seq + ['O']):
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

            if end_of_chunk(prev_tag, tag, prev_type, type_):
                chunks.append((prev_type, begin_offset, i - 1))
            if start_of_chunk(prev_tag, tag, prev_type, type_):
                begin_offset = i
            prev_tag = tag
            prev_type = type_

        return chunks


    """Compute the F1 score.

            The F1 score can be interpreted as a weighted average of the precision and
            recall, where an F1 score reaches its best value at 1 and worst score at 0.
            The relative contribution of precision and recall to the F1 score are
            equal. The formula for the F1 score is::

                F1 = 2 * (precision * recall) / (precision + recall)

            Args:
                y_true : 2d array. Ground truth (correct) target values.
                y_pred : 2d array. Estimated targets as returned by a tagger.

            Returns:
                score : float.

            Example:
                y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
                y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
                f1_score(y_true, y_pred)
                0.50
            """

    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    if mode == 'dev':
        return score
    else:
        f_score = {}
        for label in ['address', 'book', 'company', 'game', 'government', 'movie','name', 'organization', 'position', 'scene']:
            true_entities_label = set()
            pred_entities_label = set()
            for t in true_entities:
                if t[0] == label:
                    true_entities_label.add(t)
            for p in pred_entities:
                if p[0] == label:
                    pred_entities_label.add(p)
            nb_correct_label = len(true_entities_label & pred_entities_label)
            nb_pred_label = len(pred_entities_label)
            nb_true_label = len(true_entities_label)

            p_label = nb_correct_label / nb_pred_label if nb_pred_label > 0 else 0
            r_label = nb_correct_label / nb_true_label if nb_true_label > 0 else 0
            score_label = 2 * p_label * r_label / (p_label + r_label) if p_label + r_label > 0 else 0
            f_score[label] = score_label
        return f_score, score


if __name__ == '__main__':

    model_id = "qwen/Qwen2-1.5B-Instruct"
    model_dir = "./qwen/Qwen2-1___5B-Instruct"

    # 在modelscope上下载Qwen模型到本地目录下
    model_dir = snapshot_download(model_id, cache_dir="./", revision="master")

    # Transformers加载模型权重
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    # 加载、处理数据集和测试集
    train_dataset_path = "./data/train.npz"
    train_jsonl_new_path = "./data/train.jsonl"

    if not os.path.exists(train_jsonl_new_path):
        dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)

    # 得到训练集
    train_total_df = pd.read_json(train_jsonl_new_path, lines=True)
    train_total_df_len = len(train_total_df)
    train_df = train_total_df[0:train_total_df_len]
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
        experiment_name="Qwen2-0.5B-Instruct",
        description="使用通义千问Qwen2-0.5B-Instruct模型在NER数据集上微调，实现关键实体识别任务,并计算f1值",
        config={
            "model": model_id,
            "model_dir": model_dir,
            "dataset": "qgyd2021/chinese_ner_sft",
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

    # 得到测试集
    test_dataset_path = "./data/test.npz"
    test_jsonl_new_path = "./data/test.jsonl"

    if not os.path.exists(test_jsonl_new_path):
        dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)
    test_total_df = pd.read_json(test_jsonl_new_path, lines=True)
    test_total_df_len = len(test_total_df)
    test_df = test_total_df[0:test_total_df_len]

    # 加载检查点模型
    # model = AutoModelForCausalLM.from_pretrained('./output/Qwen2-NER/checkpoint-1342', device_map="auto", torch_dtype=torch.bfloat16)

    responses = []
    test_text_list = []
    for index, row in test_df.iterrows():

        instruction = row['instruction']
        input_value = row['input']
        ture_label = row['output']

        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]

        response = predict(messages, model, tokenizer)
        responses.append({
            "instruction": instruction,
            "input": input_value,
            "true_label": ture_label,
            "output": response
        })
        messages.append({"role": "assistant", "content": f"{response}"})
        result_text = f"{messages[0]}\n\n{messages[1]}\n\n{ture_label}\n\n{messages[2]}"
        test_text_list.append(swanlab.Text(result_text, caption=response))

    # 计算f1值
    import pickle

    with open('my_variable.pkl', 'wb') as file:
        pickle.dump(responses, file)
    with open('my_variable.pkl', 'rb') as f:
        a = pickle.load(f)
        data = np.load('./data/test.npz', allow_pickle=True)
        words = data['words']
        labels = data['labels']
        pre_labels = []
        tru_labels = []
        for i in range(len(a)):
            tru_label = labels[i]
            pre_label = []
            d = ast.literal_eval(json.loads(a[i]['output'])['output_list'])
            for j in range(len(d)):
                pre_label.append(d[j][2])
            pre_labels.append(pre_label)
            tru_labels.append(tru_label)
        e = str(f1_score(tru_labels, pre_labels, mode='test'))
        print(e)
        e = swanlab.Text(e, caption="F1")

    swanlab.log({"F1": e})
    swanlab.log({"Prediction": test_text_list})
    swanlab.finish()