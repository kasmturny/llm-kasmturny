import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logging.basicConfig(level=logging.ERROR)
# from transformers import TFBertPreTrainedModel,TFBertMainLayer,BertTokenizer
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

from pyorigin.core.base_agent import Bert


class MultiClassifyCn:
    def __init__(self, data_path, labels):
        self.bert = Bert()
        self.labels = labels
        self.data_path = data_path
        self.model = TFBertForSequenceClassification.from_pretrained(self.bert.bert_name, num_labels=len(self.labels))
    #


    def data_processed(self):
        # read data
        df_raw = pd.read_csv(self.data_path, sep="\t", header=None, names=["text", "label"])
        # transfer label
        df_label = pd.DataFrame(
            {"label": self.labels,
             "y": list(range(10))})
        df_raw = pd.merge(df_raw, df_label, on="label", how="left")
        train_set, x = train_test_split(df_raw,
                                        stratify=df_raw['label'],
                                        test_size=0.1,
                                        random_state=42)
        val_set, test_set = train_test_split(x,
                                             stratify=x['label'],
                                             test_size=0.5,
                                             random_state=43)
        return train_set, val_set, test_set

    def encode_examples(self, ds, limit=-1):
        def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
            return {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_masks,
            }, label

        # prepare list, so that we can build up final TensorFlow dataset from slices.
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        label_list = []
        if (limit > 0):
            ds = ds.take(limit)

        for index, row in ds.iterrows():
            review = row["text"]
            label = row["y"]
            bert_input = self.bert.tokenizer.encode_plus(review,
                                 add_special_tokens=True,  # add [CLS], [SEP]
                                 max_length=max_length,  # max length of the text that can go to BERT
                                 padding='max_length',  # add [PAD] tokens
                                 return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                 truncation=True
                                 )

            input_ids_list.append(bert_input['input_ids'])
            token_type_ids_list.append(bert_input['token_type_ids'])
            attention_mask_list.append(bert_input['attention_mask'])
            label_list.append([label])
        return tf.data.Dataset.from_tensor_slices(
            (input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)




if __name__ == '__main__':
    MultiClassifyCn=MultiClassifyCn('multi_data.txt',
                                    ["财经", "房产", "股票", "教育", "科技", "社会", "时政", "体育", "游戏", "娱乐"])
    # 参数
    max_length = 32
    batch_size = 128
    learning_rate = 2e-5
    number_of_epochs = 8
    # 数据预处理
    train_data, val_data, test_data = MultiClassifyCn.data_processed()
    # 数据编码
    ds_train_encoded = MultiClassifyCn.encode_examples(train_data).shuffle(10000).batch(batch_size)
    ds_val_encoded = MultiClassifyCn.encode_examples(val_data).batch(batch_size)
    ds_test_encoded = MultiClassifyCn.encode_examples(test_data).batch(batch_size)
    # 模型训练
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08, clipnorm=1)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    MultiClassifyCn.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    bert_history = MultiClassifyCn.model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_val_encoded)
    # 模型评估
    print("# evaluate test_set:", MultiClassifyCn.model.evaluate(ds_test_encoded))
