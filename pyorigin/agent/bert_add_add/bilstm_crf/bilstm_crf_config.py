import os
import torch


class BilstmCrfConfig:
    def __init__(self):
        # 设备
        self.device = self.cpu_or_gpu()
        self.gpu = ''
        # 数据
        self.data_dir = os.getcwd() + '\\data\\'
        self.train_dir = self.data_dir + 'train.npz'
        self.test_dir = self.data_dir + 'test.npz'
        self.files = ['train', 'test']
        self.vocab_path = self.data_dir + 'vocab.npz'
        self.dev_split_size = 0.1

        # 模型
        # self.bert_path = 'D:\\Exploitation\\All\\llm-kasmturny\\model\\bert-crf\\model\\bert-base-chinese\\' # 没有训练的模型
        # self.model_dir = 'D:\\Exploitation\\All\\llm-kasmturny\\model\\bert-crf\\experiments\\'    # 训练之后的模型
        # self.model_dir = 'C:\\Users\\wzzsa\\.cache\\huggingface\\hub\\bert_crf\\'  # 训练之后的模型
        self.exp_dir = 'D:\\Exploitation\\All\\llm-kasmturny\\model\\bilstm-crf\\'
        self.model_dir = self.exp_dir + 'model.pth'


        # 其他
        self.log_dir = os.getcwd() + '\\train.log'
        self.case_dir = os.getcwd() + '\\bad_case.txt'

        # 参数

        self.max_vocab_size = 1000000

        self.n_split = 5
        self.dev_split_size = 0.1
        self.batch_size = 32
        self.embedding_size = 128
        self.hidden_size = 384
        self.drop_out = 0.5
        self.lr = 0.001
        self.betas = (0.9, 0.999)
        self.lr_step = 5
        self.lr_gamma = 0.8

        self.epoch_num = 30
        self.min_epoch_num = 5
        self.patience = 0.0002
        self.patience_num = 5

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
    config = BilstmCrfConfig()
    print('断点')


