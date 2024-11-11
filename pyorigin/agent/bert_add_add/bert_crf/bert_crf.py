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


from pyorigin.agent.bert_add_add.bert_crf.bert_crf_config import BertCrfConfig
config = BertCrfConfig()
from pyorigin.utils import logo_util

class NERDataset(Dataset):
    """NER数据集类"""
    def __init__(self, words, labels, config, word_pad_idx=0, label_pad_idx=-1):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
        self.label2id = config.label2id
        self.id2label = {_id: _label for _label, _id in list(config.label2id.items())}
        self.dataset = self.preprocess(words, labels)
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx
        self.device = config.device

    def preprocess(self, origin_sentences, origin_labels):
        """
        Maps tokens and tags to their indices and stores them in the dict data.
        examples:
            word:['[CLS]', '浙', '商', '银', '行', '企', '业', '信', '贷', '部']
            sentence:([101, 3851, 1555, 7213, 6121, 821, 689, 928, 6587, 6956],
                        array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]))
            label:[3, 13, 13, 13, 0, 0, 0, 0, 0]
        """
        data = []
        sentences = []
        labels = []
        for line in origin_sentences:
            # replace each token by its index
            # we can not use encode_plus because our sentences are aligned to labels in list type
            words = []
            word_lens = []
            for token in line:
                words.append(self.tokenizer.tokenize(token))
                word_lens.append(len(token))
            # 变成单个字的列表，开头加上[CLS]
            words = ['[CLS]'] + [item for token in words for item in token]
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
            sentences.append((self.tokenizer.convert_tokens_to_ids(words), token_start_idxs))
        for tag in origin_labels:
            label_id = [self.label2id.get(t) for t in tag]
            labels.append(label_id)
        for sentence, label in zip(sentences, labels):
            data.append((sentence, label))
        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)

    def collate_fn(self, batch):
        """
        process batch data, including:
            1. padding: 将每个batch的data padding到同一长度（batch中最长的data长度）
            2. aligning: 找到每个sentence sequence里面有label项，文本与label对齐
            3. tensor：转化为tensor
        """
        sentences = [x[0] for x in batch]
        labels = [x[1] for x in batch]

        # batch length
        batch_len = len(sentences)

        # compute length of longest sentence in batch
        max_len = max([len(s[0]) for s in sentences])
        max_label_len = 0

        # padding data 初始化
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))
        batch_label_starts = []

        # padding and aligning
        for j in range(batch_len):
            cur_len = len(sentences[j][0])
            batch_data[j][:cur_len] = sentences[j][0]
            # 找到有标签的数据的index（[CLS]不算）
            label_start_idx = sentences[j][-1]
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)
            max_label_len = max(int(sum(label_starts)), max_label_len)

        # padding label
        batch_labels = self.label_pad_idx * np.ones((batch_len, max_label_len))
        for j in range(batch_len):
            cur_tags_len = len(labels[j])
            batch_labels[j][:cur_tags_len] = labels[j]

        # convert data to torch LongTensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_label_starts = numpy.array(batch_label_starts)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        # shift tensors to GPU if available
        batch_data, batch_label_starts = batch_data.to(self.device), batch_label_starts.to(self.device)
        batch_labels = batch_labels.to(self.device)
        return [batch_data, batch_label_starts, batch_labels]
class BertNER(BertPreTrainedModel):
    """Bert_NER模型类，实例化了一个CRF"""
    def __init__(self, config):
        super(BertNER, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        # 去除[CLS]标签等位置，获得与label对齐的pre_label表示
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output)
        # 得到判别值
        logits = self.classifier(padded_sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs

        # contain: (loss), scores
        return outputs
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
class Train_And_Test:
    def __init__(self):
        self.metrics = Metrics()


    def train(self,train_loader, dev_loader, model, optimizer, scheduler, model_dir):
        """train the model and test model performance"""
        # reload weights from restore_dir if specified
        if model_dir is not None and config.load_before:
            model = BertNER.from_pretrained(model_dir)
            model.to(config.device)
            logging.info("--------Load model from {}--------".format(model_dir))
        best_val_f1 = 0.0
        patience_counter = 0
        # start training
        for epoch in range(1, config.epoch_num + 1):
            self.train_epoch(train_loader, model, optimizer, scheduler, epoch)
            val_metrics = self.evaluate(dev_loader, model, mode='dev')
            val_f1 = val_metrics['f1']
            logging.info("Epoch: {}, dev loss: {}, f1 score: {}".format(epoch, val_metrics['loss'], val_f1))
            improve_f1 = val_f1 - best_val_f1
            if improve_f1 > 1e-5:
                best_val_f1 = val_f1
                model.save_pretrained(model_dir)
                logging.info("--------Save best model!--------")
                if improve_f1 < config.patience:
                    patience_counter += 1
                else:
                    patience_counter = 0
            else:
                patience_counter += 1
            # Early stopping and logging best f1
            if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
                logging.info("Best val f1: {}".format(best_val_f1))
                break
        logging.info("Training Finished!")

    def train_epoch(self, train_loader, model, optimizer, scheduler, epoch):
        """
        首先要得到loss,然后loss.backward()计算梯度,然后利用optimizer.step()调整权重
        """
        # set model to training mode
        model.train()
        # step number in one epoch: 336
        train_losses = 0
        for idx, batch_samples in enumerate(tqdm(train_loader)):
            batch_data, batch_token_starts, batch_labels = batch_samples
            batch_masks = batch_data.gt(0)  # get padding mask
            # compute model output and loss
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
            train_losses += loss.item()
            # clear previous gradients, compute gradients of all variables wrt loss
            model.zero_grad()
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
            # performs updates using calculated gradients
            optimizer.step()
            scheduler.step()
        train_loss = float(train_losses) / len(train_loader)
        logging.info("Epoch: {}, train loss: {}".format(epoch, train_loss))

    def evaluate(self, dev_loader, model, mode='dev'):
        # set model to evaluation mode
        model.eval()
        if mode == 'test':
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True, skip_special_tokens=True)
        id2label = config.id2label
        true_tags = []
        pred_tags = []
        sent_data = []
        dev_losses = 0

        with torch.no_grad():
            for idx, batch_samples in enumerate(dev_loader):
                batch_data, batch_token_starts, batch_tags = batch_samples
                if mode == 'test':
                    sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                                       if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
                batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
                label_masks = batch_tags.gt(-1)  # get padding mask, gt(x): get index greater than x
                # compute model output and loss
                loss = model((batch_data, batch_token_starts),
                             token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]
                dev_losses += loss.item()
                # (batch_size, max_len, num_labels)
                batch_output = model((batch_data, batch_token_starts),
                                     token_type_ids=None, attention_mask=batch_masks)[0]
                # (batch_size, max_len - padding_label_len)
                batch_output = model.crf.decode(batch_output, mask=label_masks)
                # (batch_size, max_len)
                batch_tags = batch_tags.to('cpu').numpy()
                pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
                # (batch_size, max_len - padding_label_len)
                true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])

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
        metrics['loss'] = float(dev_losses) / len(dev_loader)
        return metrics

    def test(self):
        data = np.load(config.test_dir, allow_pickle=True)
        word_test = data["words"]
        label_test = data["labels"]
        test_dataset = NERDataset(word_test, label_test, config)
        logging.info("--------Dataset Build!--------")
        # build data_loader
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                                 shuffle=False, collate_fn=test_dataset.collate_fn)
        logging.info("--------Get Data-loader!--------")
        # Prepare model
        if config.model_dir is not None:
            model = BertNER.from_pretrained(config.model_dir)
            model.to(config.device)
            logging.info("--------Load model from {}--------".format(config.model_dir))
        else:
            logging.info("--------No model to test !--------")
            return
        val_metrics = self.evaluate(test_loader, model, mode='test')
        val_f1 = val_metrics['f1']
        logging.info("test loss: {}, f1 score: {}".format(val_metrics['loss'], val_f1))
        val_f1_labels = val_metrics['f1_labels']
        for label in config.labels:
            logging.info("f1 score of {}: {}".format(label, val_f1_labels[label]))

class BertCrf:
    def __init__(self):
        pass

    def load_dev(self, mode):
        """
        mode=='train'时，加载训练集，并划分出验证集,word_train, word_dev, label_train, label_dev
        mode=='test'时，加载训练集和测试集，word_train, word_test, label_train, label_test
        """
        if mode == 'train':
            # 分离出验证集
            data = np.load(config.train_dir, allow_pickle=True)
            words = data["words"]
            labels = data["labels"]
            x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=config.dev_split_size, random_state=0)
            return x_train, x_dev, y_train, y_dev
            # 对npz文件动手

        elif mode == 'test':
            train_data = np.load(config.train_dir, allow_pickle=True)
            dev_data = np.load(config.test_dir, allow_pickle=True)
            word_train = train_data["words"]
            label_train = train_data["labels"]
            word_dev = dev_data["words"]
            label_dev = dev_data["labels"]
        else:
            word_train = None
            label_train = None
            word_dev = None
            label_dev = None
        return word_train, word_dev, label_train, label_dev

    def optimizer_grouped_parameters(self, model):
        if config.full_fine_tuning:
            # model.named_parameters(): [bert, classifier, crf]
            bert_optimizer = list(model.bert.named_parameters())
            classifier_optimizer = list(model.classifier.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': config.weight_decay},
                {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0},
                {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
                 'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
                {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
                 'lr': config.learning_rate * 5, 'weight_decay': 0.0},
                {'params': model.crf.parameters(), 'lr': config.learning_rate * 5}
            ]
        # only fine-tune the head classifier
        else:
            param_optimizer = list(model.classifier.named_parameters())
            optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
        return optimizer_grouped_parameters



if __name__ == "__main__":
    logo_util.set_logger(config.log_dir)
    # """准备数据集"""
    # word_train, word_dev, label_train, label_dev = BertCrf().load_dev('train')
    # train_dataset = NERDataset(word_train, label_train, config)
    # dev_dataset = NERDataset(word_dev, label_dev, config)
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
    #                           shuffle=True, collate_fn=train_dataset.collate_fn)
    # dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
    #                         shuffle=True, collate_fn=dev_dataset.collate_fn)
    # logging.info("————————数据集准备完成————————————")
    # """准备模型"""
    # model = BertNER.from_pretrained('bert-base-chinese', num_labels=len(config.label2id)).to(config.device)
    # logging.info("————————模型初始化完成————————————")
    # """准备优化器"""
    # optimizer_grouped_parameters=BertCrf().optimizer_grouped_parameters(model)
    # optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False, no_deprecation_warning=True)
    # train_steps_per_epoch = len(train_dataset) // config.batch_size
    # scheduler = get_cosine_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
    #                                             num_training_steps=config.epoch_num * train_steps_per_epoch)
    # logging.info("————————优化器准备完成————————————")
    # """训练模型"""
    # logging.info("————————准备开始训练————————————")
    # Train_And_Test().train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)
    # logging.info("————————测试训练模型————————————")
    Train_And_Test().test()
    print('断点')