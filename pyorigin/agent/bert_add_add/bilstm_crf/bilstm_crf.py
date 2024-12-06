import os

import numpy

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import logging
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer, BertPreTrainedModel, BertModel
from sklearn.model_selection import train_test_split
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW

import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

import logging
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from pyorigin.agent.bert_add_add.bilstm_crf.bilstm_crf_config import BilstmCrfConfig
config = BilstmCrfConfig()
from pyorigin.utils import log_util

class NERDataset(Dataset):
    """NER数据集类"""
    def __init__(self, words, labels, vocab, label2id):
        self.vocab = vocab
        self.dataset = self.preprocess(words, labels)
        self.label2id = label2id

    def preprocess(self, words, labels):
        """convert the data to ids"""
        processed = []
        for (word, label) in zip(words, labels):
            word_id = [self.vocab.word_id(w_) for w_ in word]
            label_id = [self.vocab.label_id(l_) for l_ in label]
            processed.append((word_id, label_id))
        logging.info("-------- Process Done! --------")
        return processed

    def __getitem__(self, idx):
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        return len(self.dataset)

    def get_long_tensor(self, texts, labels, batch_size):

        token_len = max([len(x) for x in texts])
        text_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        label_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        mask_tokens = torch.ByteTensor(batch_size, token_len).fill_(0)

        for i, s in enumerate(zip(texts, labels)):
            text_tokens[i, :len(s[0])] = torch.LongTensor(s[0])
            label_tokens[i, :len(s[1])] = torch.LongTensor(s[1])
            mask_tokens[i, :len(s[0])] = torch.tensor([1] * len(s[0]), dtype=torch.uint8)

        return text_tokens, label_tokens, mask_tokens

    def collate_fn(self, batch):

        texts = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        lens = [len(x) for x in texts]
        batch_size = len(batch)

        input_ids, label_ids, input_mask = self.get_long_tensor(texts, labels, batch_size)

        return [input_ids, label_ids, input_mask, lens]
class BiLSTM_CRF(nn.Module):
    """Bert_NER模型类，实例化了一个CRF"""

    def __init__(self, embedding_size, hidden_size, vocab_size, target_size, drop_out):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_size = hidden_size
        # nn.Embedding: parameter size (num_words, embedding_dim)
        # for every word id, output a embedding for this word
        # input size: N x W, N is batch size, W is max sentence len
        # output size: (N, W, embedding_dim), embedding all the words
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=drop_out,
            bidirectional=True
        )
        self.classifier = nn.Linear(hidden_size * 2, target_size)
        # https://pytorch-crf.readthedocs.io/en/stable/_modules/torchcrf.html
        self.crf = CRF(target_size, batch_first=True)

    def forward(self, inputs_ids):
        embeddings = self.embedding(inputs_ids)
        sequence_output, _ = self.bilstm(embeddings)
        tag_scores = self.classifier(sequence_output)
        return tag_scores

    def forward_with_crf(self, input_ids, input_mask, input_tags):
        tag_scores = self.forward(input_ids)
        loss = self.crf(tag_scores, input_tags, input_mask) * (-1)
        return tag_scores, loss
class Metrics:
    def __init__(self):
        pass

    def get_entities(self,seq):
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

            if self.end_of_chunk(prev_tag, tag, prev_type, type_):
                chunks.append((prev_type, begin_offset, i - 1))
            if self.start_of_chunk(prev_tag, tag, prev_type, type_):
                begin_offset = i
            prev_tag = tag
            prev_type = type_

        return chunks

    def end_of_chunk(self,prev_tag, tag, prev_type, type_):
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

    def start_of_chunk(self,prev_tag, tag, prev_type, type_):
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

    def f1_score(self,y_true, y_pred, mode='dev'):
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
        true_entities = set(self.get_entities(y_true))
        pred_entities = set(self.get_entities(y_pred))
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
            for label in config.labels:
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

    def bad_case(self,y_true, y_pred, data):
        if not os.path.exists(config.case_dir):
            os.system(r"touch {}".format(config.case_dir))  # 调用系统命令行来创建文件
        output = open(config.case_dir, 'w',encoding='utf-8')
        for idx, (t, p) in enumerate(zip(y_true, y_pred)):
            if t == p:
                continue
            else:
                output.write("bad case " + str(idx) + ": \n")
                output.write("sentence: " + str(data[idx]) + "\n")
                output.write("golden label: " + str(t) + "\n")
                output.write("model pred: " + str(p) + "\n")
        logging.info("--------Bad Cases reserved !--------")
class Vocabulary:
    """
    构建词表
    """
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.files = config.files
        self.vocab_path = config.vocab_path
        self.max_vocab_size = config.max_vocab_size
        self.word2id = {}
        self.id2word = None
        self.label2id = config.label2id
        self.id2label = config.id2label

    def __len__(self):
        return len(self.word2id)

    def vocab_size(self):
        return len(self.word2id)

    def label_size(self):
        return len(self.label2id)

    # 获取词的id
    def word_id(self, word):
        return self.word2id[word]

    # 获取id对应的词
    def id_word(self, idx):
        return self.id2word[idx]

    # 获取label的id
    def label_id(self, word):
        return self.label2id[word]

    # 获取id对应的词
    def id_label(self, idx):
        return self.id2label[idx]

    def get_vocab(self):
        """
        进一步处理，将word和label转化为id
        word2id: dict,每个字对应的序号
        idx2word: dict,每个序号对应的字
        保存为二进制文件
        """
        # 如果有处理好的，就直接load
        if os.path.exists(self.vocab_path):
            data = np.load(self.vocab_path, allow_pickle=True)
            # '[()]'将array转化为字典
            self.word2id = data["word2id"][()]
            self.id2word = data["id2word"][()]
            logging.info("-------- Vocabulary Loaded! --------")
            return
        # 如果没有处理好的二进制文件，就处理原始的npz文件
        word_freq = {}
        for file in self.files:
            data = np.load(self.data_dir + str(file) + '.npz', allow_pickle=True)
            word_list = data["words"]
            # 常见的单词id最小
            for line in word_list:
                for ch in line:
                    if ch in word_freq:
                        word_freq[ch] += 1
                    else:
                        word_freq[ch] = 1
        index = 0
        sorted_word = sorted(word_freq.items(), key=lambda e: e[1], reverse=True)
        # 构建word2id字典
        for elem in sorted_word:
            self.word2id[elem[0]] = index
            index += 1
            if index >= self.max_vocab_size:
                break
        # id2word保存
        self.id2word = {_idx: _word for _word, _idx in list(self.word2id.items())}
        # 保存为二进制文件
        np.savez_compressed(self.vocab_path, word2id=self.word2id, id2word=self.id2word)
        logging.info("-------- Vocabulary Build! --------")
class Train_And_Test:
    def __init__(self):
        self.metrics = Metrics()
        self.print_array()

    # 打印完整的numpy array
    def print_array(self):
        np.set_printoptions(threshold=np.inf)


    def epoch_train(self,train_loader, model, optimizer, scheduler, device, epoch, kf_index=0):
        """
                一样的，首先要得到loss,然后loss.backward()计算梯度,然后利用optimizer.step()调整权重
                """
        # set model to training mode
        model.train()
        # step number in one epoch: 336
        train_loss = 0.0
        for idx, batch_samples in enumerate(tqdm(train_loader)):
            x, y, mask, lens = batch_samples
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            model.zero_grad()
            tag_scores, loss = model.forward_with_crf(x, mask, y)
            train_loss += loss.item()
            # 梯度反传
            loss.backward()
            # 优化更新
            optimizer.step()
            optimizer.zero_grad()
        # scheduler
        scheduler.step()
        train_loss = float(train_loss) / len(train_loader)
        if kf_index == 0:
            logging.info("epoch: {}, train loss: {}".format(epoch, train_loss))
        else:
            logging.info("Kf epoch: {}, epoch: {}, train loss: {}".format(kf_index, epoch, train_loss))

    def train(self,train_loader, dev_loader, vocab, model, optimizer, scheduler, device, kf_index=0):
        """train the model and test model performance"""
        best_val_f1 = 0.0
        patience_counter = 0
        # start training
        for epoch in range(1, config.epoch_num + 1):
            self.epoch_train(train_loader, model, optimizer, scheduler, device, epoch, kf_index)
            with torch.no_grad():
                # dev loss calculation
                metric = self.dev(dev_loader, vocab, model, device)
                val_f1 = metric['f1']
                dev_loss = metric['loss']
                if kf_index == 0:
                    logging.info("epoch: {}, f1 score: {}, "
                                 "dev loss: {}".format(epoch, val_f1, dev_loss))
                else:
                    logging.info("Kf epoch: {}, epoch: {}, f1 score: {}, "
                                 "dev loss: {}".format(kf_index, epoch, val_f1, dev_loss))
                improve_f1 = val_f1 - best_val_f1
                if improve_f1 > 1e-5:
                    best_val_f1 = val_f1
                    torch.save(model, config.model_dir)
                    logging.info("--------Save best model!--------")
                    if improve_f1 < config.patience:
                        patience_counter += 1
                    else:
                        patience_counter = 0
                else:
                    patience_counter += 1
                # Early stopping and logging best f1
                if (
                        patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
                    logging.info("Best val f1: {}".format(best_val_f1))
                    break
        logging.info("Training Finished!")

    def sample_test(self,test_input, test_label, model, device):
        """test model performance on a specific sample"""
        test_input = test_input.to(device)
        tag_scores = model.forward(test_input)
        labels_pred = model.crf.decode(tag_scores)
        logging.info("test_label: ".format(test_label))
        logging.info("labels_pred: ".format(labels_pred))

    def dev(self,data_loader, vocab, model, device, mode='dev'):
        """test model performance on dev-set"""
        model.eval()
        true_tags = []
        pred_tags = []
        sent_data = []
        dev_losses = 0
        for idx, batch_samples in enumerate(data_loader):
            sentences, labels, masks, lens = batch_samples
            sent_data.extend([[vocab.id2word.get(idx.item()) for i, idx in enumerate(indices) if mask[i] > 0]
                              for (mask, indices) in zip(masks, sentences)])
            sentences = sentences.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            y_pred = model.forward(sentences)
            labels_pred = model.crf.decode(y_pred, mask=masks)
            targets = [itag[:ilen] for itag, ilen in zip(labels.cpu().numpy(), lens)]
            true_tags.extend([[vocab.id2label.get(idx) for idx in indices] for indices in targets])
            pred_tags.extend([[vocab.id2label.get(idx) for idx in indices] for indices in labels_pred])
            # 计算梯度
            _, dev_loss = model.forward_with_crf(sentences, masks, labels)
            dev_losses += dev_loss
        assert len(pred_tags) == len(true_tags)
        if mode == 'test':
            assert len(sent_data) == len(true_tags)

        # logging loss, f1 and report
        metrics = {}
        if mode == 'dev':
            f1 = self.metrics.f1_score(true_tags, pred_tags, mode)
            metrics['f1'] = f1
        else:
            self.metrics.bad_case(true_tags, pred_tags, sent_data)
            f1_labels, f1 = self.metrics.f1_score(true_tags, pred_tags, mode)
            metrics['f1_labels'] = f1_labels
            metrics['f1'] = f1
        metrics['loss'] = float(dev_losses) / len(data_loader)
        return metrics

    def test(self,dataset_dir, vocab, device, kf_index=0):
        """test model performance on the final test set"""
        data = np.load(dataset_dir, allow_pickle=True)
        word_test = data["words"]
        label_test = data["labels"]
        # build dataset
        test_dataset = NERDataset(word_test, label_test, vocab, config.label2id)
        # build data_loader
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                                 shuffle=True, collate_fn=test_dataset.collate_fn)
        # Prepare model
        if config.model_dir is not None:
            # model
            model = torch.load(config.model_dir)
            model.to(device)
            logging.info("--------Load model from {}--------".format(config.model_dir))
        else:
            logging.info("--------No model to test !--------")
            return
        metric = self.dev(test_loader, vocab, model, device, mode='test')
        f1 = metric['f1']
        test_loss = metric['loss']
        if kf_index == 0:
            logging.info("final test loss: {}, f1 score: {}".format(test_loss, f1))
            val_f1_labels = metric['f1_labels']
            for label in config.labels:
                logging.info("f1 score of {}: {}".format(label, val_f1_labels[label]))
        else:
            logging.info("Kf epoch: {}, final test loss: {}, f1 score: {}".format(kf_index, test_loss, f1))
        return test_loss, f1

class BilstmCrf:
    def __init__(self):
        self.input_array = [[1642, 1291, 40, 2255, 970, 46, 124, 1604, 1915, 547, 0, 173,
                        303, 124, 1029, 52, 20, 2839, 2, 2255, 2078, 1553, 225, 540,
                        96, 469, 1704, 0, 174, 3, 8, 728, 903, 403, 538, 668,
                        179, 27, 78, 292, 7, 134, 2078, 1029, 0, 0, 0, 0,
                        0],
                       [28, 6, 926, 72, 209, 330, 308, 167, 87, 1345, 1, 528,
                        412, 0, 584, 1, 6, 28, 326, 1, 361, 342, 3256, 17,
                        19, 1549, 3257, 131, 2, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0],
                       [6, 3, 58, 1930, 37, 407, 1068, 40, 1299, 1443, 103, 1235,
                        1040, 139, 879, 11, 124, 200, 135, 97, 1138, 1016, 402, 696,
                        337, 215, 402, 288, 10, 5, 5, 17, 0, 248, 597, 110,
                        84, 1, 135, 97, 1138, 1016, 402, 696, 402, 200, 109, 164,
                        0],
                       [174, 6, 110, 84, 3, 477, 332, 133, 66, 11, 557, 107,
                        181, 350, 0, 70, 196, 166, 50, 120, 26, 89, 66, 19,
                        564, 0, 36, 26, 48, 243, 1308, 0, 139, 212, 621, 300,
                        0, 444, 720, 4, 177, 165, 164, 2, 0, 0, 0, 0,
                        0]]

        self.label_array = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 14, 14, 14, 14, 14,
                        14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 4, 14, 14, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 11, 0, 1, 11, 11, 11, 11, 11, 0, 0, 0, 0, 8, 18, 18,
                        18, 18, 18, 18, 18, 18, 0, 0, 9, 19, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 8, 18, 18, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 18, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        self.test_input = torch.tensor(self.input_array, dtype=torch.long)
        self.test_label = torch.tensor(self.label_array, dtype=torch.long)

    def dev_split(self,dataset_dir):
        """split one dev set without k-fold"""
        data = np.load(dataset_dir, allow_pickle=True)
        words = data["words"]
        labels = data["labels"]
        x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=config.dev_split_size,
                                                          random_state=0)
        return x_train, x_dev, y_train, y_dev

    def k_fold_run(self,):
        """train with k-fold"""
        # set the logger
        logo_util.set_logger(config.log_dir)
        # 设置gpu为命令行参数指定的id
        if config.gpu != '':
            device = torch.device(f"cuda:{config.gpu}")
        else:
            device = torch.device("cpu")
        logging.info("device: {}".format(device))
        # 处理数据，分离文本和标签
        # 建立词表
        vocab = Vocabulary(config)
        vocab.get_vocab()
        # 分离出验证集
        data = np.load(config.train_dir, allow_pickle=True)
        words = data["words"]
        labels = data["labels"]
        kf = KFold(n_splits=config.n_split)
        kf_data = kf.split(words, labels)
        kf_index = 0
        total_test_loss = 0
        total_f1 = 0
        for train_index, dev_index in kf_data:
            kf_index += 1
            word_train = words[train_index]
            label_train = labels[train_index]
            word_dev = words[dev_index]
            label_dev = labels[dev_index]
            test_loss, f1 = self.run(word_train, label_train, word_dev, label_dev, vocab, device, kf_index)
            total_test_loss += test_loss
            total_f1 += f1
        average_test_loss = float(total_test_loss) / config.n_split
        average_f1 = float(total_f1) / config.n_split
        logging.info("Average test loss: {} , average f1 score: {}".format(average_test_loss, average_f1))

    def simple_run(self,):
        """train without k-fold"""
        # set the logger
        logo_util.set_logger(config.log_dir)
        # 设置gpu为命令行参数指定的id
        if config.gpu != '':
            device = torch.device(f"cuda:{config.gpu}")
        else:
            device = torch.device("cpu")
        logging.info("device: {}".format(device))
        # 处理数据，分离文本和标签
        # 建立词表
        vocab = Vocabulary(config)
        vocab.get_vocab()
        # 分离出验证集
        word_train, word_dev, label_train, label_dev = self.dev_split(config.train_dir)
        # simple run without k-fold
        self.run(word_train, label_train, word_dev, label_dev, vocab, device)

    def run(self,word_train, label_train, word_dev, label_dev, vocab, device, kf_index=0):
        # build dataset
        train_dataset = NERDataset(word_train, label_train, vocab, config.label2id)
        dev_dataset = NERDataset(word_dev, label_dev, vocab, config.label2id)
        # build data_loader
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  shuffle=True, collate_fn=train_dataset.collate_fn)
        dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                                shuffle=True, collate_fn=dev_dataset.collate_fn)
        # model
        model = BiLSTM_CRF(embedding_size=config.embedding_size,
                           hidden_size=config.hidden_size,
                           drop_out=config.drop_out,
                           vocab_size=vocab.vocab_size(),
                           target_size=vocab.label_size())
        model.to(device)
        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=config.betas)
        scheduler = StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_gamma)
        # how to initialize these parameters elegantly
        for p in model.crf.parameters():
            _ = torch.nn.init.uniform_(p, -1, 1)
        # train and test
        Train_And_Test().train(train_loader, dev_loader, vocab, model, optimizer, scheduler, device, kf_index)
        with torch.no_grad():
            # test on the final test set
            test_loss, f1 = Train_And_Test().test(config.test_dir, vocab, device, kf_index)
            # sample_test(test_input, test_label, model, device)
        return test_loss, f1

if __name__ == "__main__":
    BilstmCrf().simple_run()