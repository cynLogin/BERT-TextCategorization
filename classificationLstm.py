import csv

import jieba
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import random
import time
import datetime
from collections import Counter
from itertools import chain
# 加载预训练词嵌入模型示例
from gensim.models import Word2Vec

labels_map = {
    "体育": 0,
    "娱乐": 1,
    "家居": 2,
    "房产": 3,
    "教育": 4,
    "时尚": 5,
    "时政": 6,
    "游戏": 7,
    "科技": 8,
    "财经": 9
#     "Art": 0,
#     "Literature": 1,
#     "Education": 2,
#     "Philosophy": 3,
#     "History": 4,
#     "Space": 5,
#     "Energy": 6,
#     "Electronics": 7,
#     "Communication": 8,
#     "Computer": 9,
# "Mine": 10,
# "Transport": 11,
# "Enviornment": 12,
#     "Agriculture": 13,
# "Economy": 14,
# "Law": 15,
# "Medical": 16,
#     "Military": 17,
# "Politics": 18,
# "Sports": 19,
}
xunlian_dir="./文本分类（新闻语料）/cnews.train.txt"
ceshi_dir="./文本分类（新闻语料）/cnews.test.txt"
# 设置LSTM模型参数
embedding_dim = 200  # 修改词嵌入维度
hidden_dim = 256  # 修改隐藏层维度
vocab_size = 20000  # 增加词汇表大小
num_labels = 10
# GPU检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_vocab(data, vocab_size):
    counter = Counter(chain(*[jieba.cut(text) for text in data]))
    most_common = counter.most_common(vocab_size - 1)  # 保留一个位置给未知词汇
    word_to_idx = {word: i+1 for i, (word, _) in enumerate(most_common)}  # 从1开始索引
    word_to_idx["<UNK>"] = 0  # 为未知词汇添加索引0
    return word_to_idx

# 函数：格式化时间显示
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def encode_text(text, word_to_idx, max_length):
    words = list(jieba.cut(text))
    return [word_to_idx.get(word, 0) for word in words[:max_length]]  # 使用0替换未知词汇


# 加载停用词
stopwords = set()
with open("stopwords.txt", "r", encoding="utf-8") as f:
    for line in f:
        stopwords.add(line.strip())

# 加载训练和测试数据
def load_data(file_path):
    data, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                data.append(parts[1])
                labels.append(int(labels_map[parts[0]]))
    return data, labels

train_data, train_labels = load_data(xunlian_dir)
test_data, test_labels = load_data(ceshi_dir)

# LSTM模型
class LSTMForSequenceClassification(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_labels):
        super(LSTMForSequenceClassification, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_labels)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids):
        embeddings = self.word_embeddings(input_ids)
        lstm_out, _ = self.lstm(embeddings)
        out = self.fc(lstm_out[:, -1, :])
        return self.softmax(out)




model = LSTMForSequenceClassification(embedding_dim, hidden_dim, vocab_size, num_labels).to(device)

model.to(device)  # 移动模型到GPU


# 数据预处理
def preprocess_data(data, labels, word_to_idx, max_length=64):
    input_ids, processed_labels = [], []

    for text, label in zip(data, labels):
        text = "".join([word for word in jieba.cut(text) if word not in stopwords])
        encoded_text = encode_text(text, word_to_idx, max_length)
        input_ids.append(encoded_text)
        processed_labels.append(label)



    # 确保input_ids为Long类型
    input_ids = torch.tensor([np.pad(ids, (0, max_length - len(ids)), mode='constant') for ids in input_ids]).long().to(device)
    processed_labels = torch.tensor(processed_labels).long().to(device)
    return input_ids, processed_labels



# 构建词汇表和预处理数据
word_to_idx = build_vocab(train_data + test_data, vocab_size)
print("Vocabulary size:", len(word_to_idx))

# 使用预训练的Word2Vec词嵌入模型
from gensim.models import KeyedVectors

word2vec_model= KeyedVectors.load_word2vec_format("45000-small.txt")
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, idx in word_to_idx.items():
    if word in word2vec_model:
        embedding_matrix[idx] = word2vec_model[word]


# 确保词嵌入矩阵在正确的设备上，并转换为float类型
embedding_matrix = torch.tensor(embedding_matrix).float().to(device)
model.word_embeddings.weight = nn.Parameter(embedding_matrix, requires_grad=False)
train_inputs, train_labels = preprocess_data(train_data, train_labels, word_to_idx)
test_inputs, test_labels = preprocess_data(test_data, test_labels, word_to_idx)

# 创建DataLoader
batch_size = 16
train_dataset = TensorDataset(train_inputs, train_labels)
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
validation_dataset = TensorDataset(test_inputs, test_labels)
validation_dataloader = DataLoader(validation_dataset, sampler=SequentialSampler(validation_dataset), batch_size=batch_size)


# 设置优化器和调度器
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 100
total_steps = len(train_dataloader) * epochs

# 训练模型
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

loss_list=[]
for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    t0 = time.time()
    total_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)

        model.zero_grad()
        outputs = model(b_input_ids)
        loss = nn.CrossEntropyLoss()(outputs, b_labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
    loss_list.append("{0:.2f}".format(avg_train_loss))

# 评估模型
print("\nRunning Validation...")
t0 = time.time()
model.eval()
predictions, true_labels = [], []

# 评估模型
for batch in validation_dataloader:
    b_input_ids, b_labels = batch[0].to(device), batch[1].to(device)
    with torch.no_grad():
        outputs = model(b_input_ids)
    logits = outputs
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    predictions.append(logits)
    true_labels.append(label_ids)

print("  Validation took: {:}".format(format_time(time.time() - t0)))

# 计算性能指标
flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = np.concatenate(true_labels, axis=0)

precision = precision_score(flat_true_labels, flat_predictions,average='weighted')
recall = recall_score(flat_true_labels, flat_predictions,average='weighted')
f1 = f1_score(flat_true_labels, flat_predictions,average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

with open("./output/loss.csv", "w", encoding="utf-8",newline='') as f:
    csv_writer = csv.writer(f)
    for row in loss_list:
        csv_writer.writerow(row)