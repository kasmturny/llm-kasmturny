import os
import json
import time
import torch
import pickle
import random
import swanlab
import numpy as np
import pandas as pd
from tqdm import tqdm
from sqlalchemy import null
from functools import wraps
from datasets import Dataset
from datetime import datetime
from peft import LoraConfig, TaskType, get_peft_model
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed_time = end_time - start_time
        print(f"""
\033[93m
**************************************************
[{current_datetime}] 
    {{
        '函数' : \033[92m'{func.__name__}'\033[93m
        '执行时间': \033[92m'{elapsed_time} 秒'\033[93m
    }}
..................................................\033[0m
""")
        return result
    return wrapper


class ConfigureClass:
    #配置
    model_id = "qwen/Qwen2-0.5B-Instruct"
    model_dir = "./qwen/Qwen2-0___5B-Instruct"
    output_dir = "./output/Qwen2-NER"
    train_origin_path = './data/train.json'
    train_new_path = './data/train_new_1213.jsonl'
    test_origin_path = './data/test.json'
    test_new_path = './data/test_new_1213.jsonl'
    dataset_description = "CLUE 2020数据集是中文语言理解测评基准（CLUE）的一部分，它包含了多个针对中文语言理解任务的评测数据集"
    swanlab_config = {
        "project": "Qwen2-NER-fintune",
        "experiment_name": "Qwen2-0.5B-Instruct-Clue2020",
        "description": "使用通义千问Qwen2-0.5B-Instruct模型在NER数据集clue2020上微调address任务，实现关键实体识别,并返回f1值。",
        "logdir": './swanlog',
    }

    #初始化
    model_dir = snapshot_download(model_id, cache_dir="./", revision="master")  # 会下载到当前目录,下载一次之后这句不会触发，所以需要model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()  # 梯度检查点
    swanlabrun = swanlab.init(
        project=swanlab_config["project"],
        experiment_name=swanlab_config["experiment_name"],
        description=swanlab_config["description"],
        logdir=swanlab_config["logdir"],
    )
    swanlabcallback = SwanLabCallback(
        config={
            "model_id": model_id,
            "model_dir": model_dir,
            "output_dir": output_dir,
            "dataset_description": dataset_description,
        },
    )
    #TODO:todo方法,
    # 训练集和测试集是同一个数据结构，然后训练集和数据集通过todo1的规则，完成转换
    # 之后新训练集进行模型训练，拿测试集进行模型回复，加上回复的字段
    # 也就是说新测试集有四个字段，其中后面两个字段可以通过同一个规则转换到bio数据进行f1的计算，也就是todo3和todo4可以是同一个规则，也可以分别处理预测和真实数据
    # todo2则是检测输出格式，todo5则是进行模型自检
    @staticmethod
    def todo1(origin_path):
        messages = [
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},

            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
            {'instruction': '系统提示词', 'input': '用户输入', 'output': 'LLM输出'},
        ]
        return messages

    @staticmethod
    def todo2(llm_output, input_value):
        return True

    @staticmethod
    def todo3(true_text_list):
        y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        return y_true

    @staticmethod
    def todo4(pred_text_list):
        y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        return y_pred

    @staticmethod
    def todo5(response,model):
        return True

    @staticmethod
    @timing_decorator
    def configure_class():
        time.sleep(1)

    configure_class()

class DatasetClass:
    def __init__(self):
        self.configure = ConfigureClass
        self.train_dataset = None
        self.test_dataset = None
        self.input_ids_max_length = 0
        self.is_dataset_exist(ConfigureClass.train_origin_path, ConfigureClass.train_new_path,
                              ConfigureClass.test_origin_path, ConfigureClass.test_new_path)

    @timing_decorator
    def is_dataset_exist(self, train_origin_path, train_new_path, test_origin_path, test_new_path):
        if os.path.exists(train_new_path):
            os.remove(train_new_path)
        if not os.path.exists(train_new_path):
            messages = self.messages_transfer(train_origin_path)
            self.dataset_jsonl_transfer(messages, train_new_path)
        train_total_df = pd.read_json(train_new_path, lines=True)
        train_nums = len(train_total_df) #指定训练数量
        train_nums = 1000
        train_df = train_total_df[0:train_nums]
        train_ds = Dataset.from_pandas(train_df)
        train_dataset = train_ds.map(self.process_func, remove_columns=train_ds.column_names)
        self.train_dataset = train_dataset
        if os.path.exists(test_new_path):
            os.remove(test_new_path)
        if not os.path.exists(test_new_path):
            messages = self.messages_transfer(test_origin_path)
            self.dataset_jsonl_transfer(messages, test_new_path)
        test_total_df = pd.read_json(test_new_path, lines=True)
        test_nums = len(test_total_df)  # 指定测试数量
        test_nums = 20
        test_df = test_total_df[0:test_nums]
        self.test_dataset = test_df

    def messages_transfer(self, oring_path):
        #TODO: todo1将原始数据集转换为大模型微调所需数据格式的新数据集,就是一个有instruction，input，output三个字段的字典的列表messages
        messages = ConfigureClass.todo1(oring_path)
        return messages

    def dataset_jsonl_transfer(self, messages, new_path):
        # 保存重构后的JSONL文件，需要的是具有instruction，input，output三个字段的列表
        with open(new_path, "w", encoding="utf-8") as file:
            for message in messages:
                file.write(json.dumps(message, ensure_ascii=False) + "\n")

    def process_func(self,example):

        MAX_LENGTH = 384
        input_ids = []
        attention_mask = []
        labels = []
        instruction = ConfigureClass.tokenizer(
            f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        response = ConfigureClass.tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [ConfigureClass.tokenizer.pad_token_id]
        attention_mask = (
                instruction["attention_mask"] + response["attention_mask"] + [1]
        )
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [ConfigureClass.tokenizer.pad_token_id]
        if len(input_ids) > self.input_ids_max_length:
            self.input_ids_max_length = len(input_ids)
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

class TrainClass:
    def __init__(self,train_dataset):
        self.configure = ConfigureClass
        self.new_model = None
        self.train_model(train_dataset)

    @timing_decorator
    def train_model(self, train_dataset):

        #lora配置
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, #任务类型,还有其他任务，不知道能不能有效
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.new_model = get_peft_model(ConfigureClass.model, config)

        #训练配置
        args = TrainingArguments(
            output_dir=ConfigureClass.output_dir,
            per_device_train_batch_size=4,# 一个步骤四个数据
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,# 累计四个步骤更新梯度————————一次处理16个数据
            logging_steps=10,
            num_train_epochs=4,# 样本数量*n
            save_steps=100,
            learning_rate=1e-3,# 学习率
            save_on_each_node=True,
            gradient_checkpointing=True,
            report_to="none",
        )
        trainer = Trainer(
            model=self.new_model,
            args=args,
            train_dataset=train_dataset,#需要的数据集
            data_collator=DataCollatorForSeq2Seq(tokenizer=ConfigureClass.tokenizer, padding=True),
            callbacks=[ConfigureClass.swanlabcallback],
        )

        #训练
        trainer.train()
class TestClass:
    def __init__(self, model, test_dataset):
        self.configure = ConfigureClass
        self.tokenizer = ConfigureClass.tokenizer
        self.test_model = model
        self.test_dataset = test_dataset
        self.text_messages_list = []
        self.test_text_list = []
        self.f1 = 0
        self.run_test()

    @timing_decorator
    def run_test(self):
        for index, row in tqdm(self.test_dataset.iterrows(), total=len(self.test_dataset)):
            # 原始数据
            instruction = row['instruction']
            input_value = row['input']
            output_value = row['output']

            # 预测数据
            messages = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{input_value}"}
            ]
            while True:
                response = self.predict(messages)
                # TODO：todo2格式验证
                if self.check(response, input_value):
                    # TODO:todo5自我验证
                    if ConfigureClass.todo5(response, self.test_model):
                        break

            messages.append({"期望输出": f"{output_value}"})
            messages.append({"role": "assistant", "content": f"{response}"})
            self.text_messages_list.append(messages)
            result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}\n\n{messages[3]}"
            self.test_text_list.append(swanlab.Text(result_text, caption=response))
        self.f1 = self.f1_compute()
        swanlab.log({"预测结果":self.test_text_list})
        swanlab.log({"十类F1": [swanlab.Text(str(self.f1[0]),caption='f1_10')] })
        swanlab.log({"总体F1": [swanlab.Text(str(self.f1[1]), caption='f1_all')] })

    def predict(self, messages):
        device = "cuda"
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        generated_ids = self.test_model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    def check(self, llm_output, input_value):
        return ConfigureClass.todo2(llm_output, input_value)

    def f1_compute(self):
        def true_text_list_transform(true_text_list):
            #TODO:todo3真实文本信息转化为标签数据
            y_true = ConfigureClass.todo3(true_text_list)
            return y_true
        def pred_text_list_transform(pred_text_list):
            #TODO:todo4预测文本信息转化为标签数据
            y_pred = ConfigureClass.todo4(pred_text_list)
            return y_pred
        true_text_list = []
        pred_text_list = []
        for meta in self.text_messages_list:
            true_text = meta[2]['期望输出']
            pred_text = meta[3]['content']
            true_text_list.append(true_text)
            pred_text_list.append(pred_text)
        y_true = true_text_list_transform(true_text_list)
        y_pred = pred_text_list_transform(pred_text_list)
        return self.f1_score(y_true, y_pred)

    def f1_score(self,y_true, y_pred, mode='test'):
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
            for label in ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization',
                          'position', 'scene']:
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

if __name__ == "__main__":

    dataset = DatasetClass()
    train_model = TrainClass(dataset.train_dataset)
    test_model = TestClass(train_model.new_model, dataset.test_dataset)
    swanlab.finish()






































