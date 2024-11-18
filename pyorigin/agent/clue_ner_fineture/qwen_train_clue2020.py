import json
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
import numpy as np

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []
    with open(origin_path, 'r', encoding='utf-8') as file:
        # 先读取到内存中，然后逐行处理
        for line in file.readlines():
            data = json.loads(line)
            output_dict = {}
            for entity_type, entities in data["label"].items():
                for entity_text, entity_indices in entities.items():
                    entity_id = len(output_dict) + 1
                    output_dict[str(entity_id)] = {
                        "entity_text": entity_text,
                        "entity_type": entity_type,
                        "entity_index": entity_indices
                    }
            message = ({
                "instruction": """你是一个文本实体识别领域的专家,你需要对input分析 地址、书籍、公司、游戏、政府、电影、名称、组织、职位、场景 实体,返回实体文本,实体类型,实体索引,以json格式输出,如input="浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，",那么output={"1":{"entity_text": "浙商银行", "entity_type": "company", "entity_index":[0.3]},"2":{"entity_text": "叶老桂", "entity_type": "name", "entity_index":[[9, 11],[32, 34]]}},注意: 1. 输出的每一行都必须是正确的 json 字符串. 2.input的索引遵守python的[0,len(input)]，下标从0开始""",
                "input": data["text"],
                "output": json.dumps(output_dict, ensure_ascii=False)
            })
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
    system_prompt = """你是一个文本实体识别领域的专家,你需要对input分析 地址、书籍、公司、游戏、政府、电影、名称、组织、职位、场景 实体,返回实体文本,实体类型,实体索引,以json格式输出,如input="浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，",那么output={"1":{"entity_text": "浙商银行", "entity_type": "company", "entity_index":[0.3]},"2":{"entity_text": "叶老桂", "entity_type": "name", "entity_index":[[9, 11],[32, 34]]}},注意: 1. 输出的每一行都必须是正确的 json 字符串. 2.input的索引遵守python的[0,len(input)],下标从0开始."""
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

    print(response,'\n')

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

def ture_labels_f(test_nums):
    data = np.load('./data/test.npz', allow_pickle=True)
    ture_ture_labels = []
    for meta in data['labels'][0:test_nums]:
        ture_ture_labels.append(meta)
    return ture_ture_labels

def pre_labels_f(responses):
    labels = []
    for input_value, response in responses:
        pre_labels = ['O'] * len(input_value)
        response = json.loads(response)
        for key, value in response.items():
            for start_index, end_index in value['entity_index']:
                if start_index < 0 or end_index >= len(input_value):
                    labels.append(pre_labels)
                    continue
                if start_index == end_index:
                    pre_labels[start_index] = 'B-' + value['entity_type']
                else:
                    pre_labels[start_index] = 'B-' + value['entity_type']
                    pre_labels[start_index + 1:end_index + 1] = ['I-' + value['entity_type']] * (
                                len(value['entity_text']) - 1)
        labels.append(pre_labels)
    return labels


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
    # train_dataset_path = "ccfbdci.jsonl"
    # train_jsonl_new_path = "ccf_train.jsonl"
    train_dataset_path = "./data/train.json"
    train_jsonl_new_path = "./data/train_new.jsonl"

    if not os.path.exists(train_jsonl_new_path):
        dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)

    # 得到训练集
    train_total_df = pd.read_json(train_jsonl_new_path, lines=True)
    train_nums = len(train_total_df)
    train_df = train_total_df[0:train_nums]
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
        description="使用通义千问Qwen2-1.5B-Instruct模型在NER数据集clue2020上微调，实现关键实体识别任务,并返回f1值。",
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

    test_dataset_path = "./data/test.json"
    test_jsonl_new_path = "./data/test_new.jsonl"

    if not os.path.exists(test_jsonl_new_path):
        dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)



    test_total_df = pd.read_json(test_jsonl_new_path, lines=True)
    test_nums = len(test_total_df)       # 指定测试数量
    test_df = test_total_df[0:test_nums]

    model = AutoModelForCausalLM.from_pretrained(model, device_map="auto",
                                                 torch_dtype=torch.bfloat16)

    responses = []
    test_text_list = []
    for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
        instruction = row['instruction']
        input_value = row['input']
        output_value = row['output']
        print(f'input_value={input_value}', '\n', f'output_value = {output_value}', '\n', '预测结果：')

        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]

        response = predict(messages, model, tokenizer)
        responses.append((input_value, response))
        messages.append({"role": "assistant", "content": f"{response}"})
        result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
        test_text_list.append(swanlab.Text(result_text, caption=response))
    pre_labels = pre_labels_f(responses)
    ture_labels = ture_labels_f(test_nums)
    print(f1_score(pre_labels, ture_labels))
    swanlab.log({"Prediction": test_text_list})
    swanlab.log({"F1": f1_score(pre_labels, ture_labels)})
    swanlab.finish()

