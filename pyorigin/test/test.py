import os
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
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
import config

class nerdataset(Dataset):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.data_dir = 'D:\\Exploitation\\All\\llm-kasmturny\\pyorigin\\test\\'

        self.train_data, self.train_labels = self.preprocess('train')
        self.test_data, self.test_labels = self.preprocess('test')

        self.label2id = {
                        "O": 0,
                        "B-address": 1,
                        "B-book": 2,
                        "B-company": 3,
                        'B-game': 4,
                        'B-government': 5,
                        'B-movie': 6,
                        'B-name': 7,
                        'B-organization': 8,
                        'B-position': 9,
                        'B-scene': 10,
                        "I-address": 11,
                        "I-book": 12,
                        "I-company": 13,
                        'I-game': 14,
                        'I-government': 15,
                        'I-movie': 16,
                        'I-name': 17,
                        'I-organization': 18,
                        'I-position': 19,
                        'I-scene': 20,
                        "S-address": 21,
                        "S-book": 22,
                        "S-company": 23,
                        'S-game': 24,
                        'S-government': 25,
                        'S-movie': 26,
                        'S-name': 27,
                        'S-organization': 28,
                        'S-position': 29,
                        'S-scene': 30
                        }
        self.train_data_data = self.process(self.train_data, self.train_labels)
        self.test_data_data = self.process(self.test_data, self.test_labels)



    def preprocess(self, mode):
        """
        params:
            words：将json文件每一行中的文本分离出来，存储为words列表
            labels：标记文本对应的标签，存储为labels
        examples:
            words示例：['生', '生', '不', '息', 'C', 'S', 'O', 'L']
            labels示例：['O', 'O', 'O', 'O', 'B-game', 'I-game', 'I-game', 'I-game']
        """
        input_dir = self.data_dir + str(mode) + '.json'
        output_dir = self.data_dir + str(mode) + '.npz'
        if os.path.exists(output_dir) is True:
            return
        word_list = []
        label_list = []
        with open(input_dir, 'r', encoding='utf-8') as f:
            # 先读取到内存中，然后逐行处理
            for line in f.readlines():
                # loads()：用于处理内存中的json对象，strip去除可能存在的空格
                json_line = json.loads(line.strip())

                text = json_line['text']
                words = list(text)
                # 如果没有label，则返回None
                label_entities = json_line.get('label', None)
                labels = ['O'] * len(words)

                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                word_list.append(words)
                label_list.append(labels)
                # 保存成二进制文件
            logging.info("--------{} data process DONE!--------".format(mode))
            return word_list, label_list

    def process(self, origin_sentences, origin_labels):
        """
        Maps tokens and tags to their indices and stores them in the dict data.
        examples:
            word:['[CLS]', '浙', '商', '银', '行', '企', '业', '信', '贷', '部']
            sentence:(
                        [101, 3851, 1555, 7213, 6121, 821, 689, 928, 6587, 6956],
                        array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
                        )
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

    def dev_split(self):
        """split dev set"""

        words = self.train_data
        labels = self.train_labels
        x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=0.1,
                                                          random_state=0)
        return x_train, x_dev, y_train, y_dev

class NERDataSet(Dataset):
    def __init__(self, words, labels, word_pad_idx=0, label_pad_idx=-1):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
        label2id = {
                        "O": 0,
                        "B-address": 1,
                        "B-book": 2,
                        "B-company": 3,
                        'B-game': 4,
                        'B-government': 5,
                        'B-movie': 6,
                        'B-name': 7,
                        'B-organization': 8,
                        'B-position': 9,
                        'B-scene': 10,
                        "I-address": 11,
                        "I-book": 12,
                        "I-company": 13,
                        'I-game': 14,
                        'I-government': 15,
                        'I-movie': 16,
                        'I-name': 17,
                        'I-organization': 18,
                        'I-position': 19,
                        'I-scene': 20,
                        "S-address": 21,
                        "S-book": 22,
                        "S-company": 23,
                        'S-game': 24,
                        'S-government': 25,
                        'S-movie': 26,
                        'S-name': 27,
                        'S-organization': 28,
                        'S-position': 29,
                        'S-scene': 30
                        }
        self.label2id = label2id
        self.id2label = {_id: _label for _label, _id in list(label2id.items())}

        self.dataset = self.preprocess(words, labels)
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx

        self.device = torch.device("cpu")


    def preprocess(self, origin_sentences, origin_labels):
        """
        Maps tokens and tags to their indices and stores them in the dict data.
        examples:
            word:['[CLS]', '浙', '商', '银', '行', '企', '业', '信', '贷', '部']
            sentence:(
                        [101, 3851, 1555, 7213, 6121, 821, 689, 928, 6587, 6956],
                        array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
                        )
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



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """sample data to get batch"""
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

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
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        # shift tensors to GPU if available
        batch_data, batch_label_starts = batch_data.to(self.device), batch_label_starts.to(self.device)
        batch_labels = batch_labels.to(self.device)
        return [batch_data, batch_label_starts, batch_labels]


from torchcrf import CRF


class BertNER(BertPreTrainedModel):
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

def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    if model_dir is not None and config.load_before:
        model = BertNER.from_pretrained(model_dir)
        model.to(config.device)
        print("--------Load model from {}--------".format(model_dir))
        logging.info("--------Load model from {}--------".format(model_dir))
    best_val_f1 = 0.0
    patience_counter = 0
    # start training
    print('--------Start training!--------')
    for epoch in range(1, config.epoch_num + 1):
        train_epoch(train_loader, model, optimizer, scheduler, epoch)
        val_metrics = evaluate(dev_loader, model, mode='dev')
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

def train_epoch(train_loader, model, optimizer, scheduler, epoch):
    # set model to training mode
    model.train()
    # step number in one epoch: 336
    train_losses = 0
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, batch_token_starts, batch_labels = batch_samples
        batch_masks = batch_data.gt(0)  # get padding mask
        # compute model output and loss
        loss = model((batch_data, batch_token_starts), token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
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

def evaluate(dev_loader, model, mode='dev'):
    # set model to evaluation mode
    model.eval()
    if mode == 'test':
        tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True, skip_special_tokens=True)
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

if __name__ == '__main__':
    nerdataset = nerdataset()
    x_train, x_dev, y_train, y_dev = nerdataset.dev_split()
    train_dataset = NERDataSet(x_train, y_train)
    dev_dataset = NERDataSet(x_dev, y_dev)
    train_size = len(train_dataset)
    # build data_loader
    train_loader = DataLoader(train_dataset, batch_size=32,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=32,
                            shuffle=True, collate_fn=dev_dataset.collate_fn)
    model = BertNER.from_pretrained('bert-base-chinese')

    if True:
        # model.named_parameters(): [bert, classifier, crf]
        bert_optimizer = list(model.bert.named_parameters())
        classifier_optimizer = list(model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': (3e-5) * 5, 'weight_decay': 0.01},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': (3e-5) * 5, 'weight_decay': 0.0},
            {'params': model.crf.parameters(), 'lr': (3e-5) * 5}
        ]
        # only fine-tune the head classifier
    # else:
    #     param_optimizer = list(model.classifier.named_parameters())
    #     optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=(3e-5), correct_bias=False)
    train_steps_per_epoch = train_size // 32
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(50 // 10) * train_steps_per_epoch,
                                                num_training_steps=50 * train_steps_per_epoch)

    # Train the model
    logging.info("--------Start Training!--------")
    print('nihao')
    print('你好')


    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)


    print('断点')