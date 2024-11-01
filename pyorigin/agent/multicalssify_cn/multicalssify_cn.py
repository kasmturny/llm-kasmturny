import logging
logging.basicConfig(level=logging.ERROR)
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from pyorigin.core.base_agent import Bert


class MultiClassifyCn:
    def __init__(self):
        self.model = Bert().tfbert_model
        self.tokenizer = Bert().bert_tokenizer
        self.labels = ["体育", "娱乐", "财经", "汽车", "科技", "军事", "教育", "房产", "时尚", "游戏"]
        self.label2id = {label: i for i, label in enumerate(self.labels)}

    def data_preprocess(self, data):
        # 数据预处理
        # ...
        return data

if __name__ == '__main__':
    multiclassifycn = MultiClassifyCn()
    print()