import os
import torch


class BertCrfConfig:
    def __init__(self):
        # 数据
        self.data_dir = os.getcwd() + '\\data\\'
        self.train_dir = self.data_dir + 'train.npz'
        self.test_dir = self.data_dir + 'test.npz'
        self.files = ['train', 'test']
        self.dev_split_size = 0.1

        # 模型
        # self.bert_path = 'D:\\Exploitation\\All\\llm-kasmturny\\model\\bert-crf\\model\\bert-base-chinese\\' # 没有训练的模型
        # self.model_dir = 'D:\\Exploitation\\All\\llm-kasmturny\\model\\bert-crf\\experiments\\'    # 训练之后的模型
        self.model_dir = 'C:\\Users\\wzzsa\\.cache\\huggingface\\hub\\bert_softmax\\'  # 训练之后的模型

        # 其他
        self.log_dir = os.getcwd() + '\\train.log'
        self.case_dir = os.getcwd() + '\\bad_case.txt'

        # 参数
        self.full_fine_tuning = True
        self.load_before = False
        self.device = self.cpu_or_gpu()
        self.learning_rate = 3e-5
        self.weight_decay = 0.01
        self.clip_grad = 5
        self.batch_size = 32
        self.epoch_num = 50
        self.min_epoch_num = 5
        self.patience = 0.0002
        self.patience_num = 10

        # 标签
        self.labels = ['address', 'book', 'company', 'game', 'government', 'movie',
                       'name', 'organization', 'position', 'scene']
        self.label2id = {"O": 0,
                         "B-address": 1, "B-book": 2, "B-company": 3, 'B-game': 4, 'B-government': 5, 'B-movie': 6,
                         'B-name': 7, 'B-organization': 8, 'B-position': 9, 'B-scene': 10,
                         "I-address": 11, "I-book": 12, "I-company": 13, 'I-game': 14, 'I-government': 15, 'I-movie': 16,
                         'I-name': 17, 'I-organization': 18, 'I-position': 19, 'I-scene': 20,
                         "S-address": 21, "S-book": 22, "S-company": 23, 'S-game': 24, 'S-government': 25, 'S-movie': 26,
                         'S-name': 27, 'S-organization': 28, 'S-position': 29, 'S-scene': 30
        }
        self.id2label = {_id: _label for _label, _id in list(self.label2id.items())}

    def cpu_or_gpu(self):
        gpu = ''
        if gpu != '':
            self.device = torch.device(f"cuda:{gpu}")
        else:
            self.device = torch.device("cpu")
        return self.device




if __name__ == '__main__':
    config = BertCrfConfig()
    print('断点')


