
import jieba
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import random
import time
import datetime



labels_map = {
    # "体育": 0,
    # "娱乐": 1,
    # "家居": 2,
    # "房产": 3,
    # "教育": 4,
    # "时尚": 5,
    # "时政": 6,
    # "游戏": 7,
    # "科技": 8,
    # "财经": 9
"Art": 0,
    "Literature": 1,
    "Education": 2,
    "Philosophy": 3,
    "History": 4,
    "Space": 5,
    "Energy": 6,
    "Electronics": 7,
    "Communication": 8,
    "Computer": 9,
"Mine": 10,
"Transport": 11,
"Enviornment": 12,
    "Agriculture": 13,
"Economy": 14,
"Law": 15,
"Medical": 16,
    "Military": 17,
"Politics": 18,
"Sports": 19,
}
# GPU检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 函数：格式化时间显示
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

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
            parts = line.strip().split("\t",1)
            if len(parts) == 2:
                data.append(parts[1])
                labels.append(labels_map[parts[0]])
    return data, labels

train_data, train_labels = load_data("./文本分类1/训练集.txt")
test_data, test_labels = load_data("./文本分类1/测试集.txt")

proxies = {
    "http": "127.0.0.1:15732",
    "https": "127.0.0.1:15732",
}
# 设置BERT
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name,proxies=proxies)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=20,proxies=proxies)
model.to(device)  # 移动模型到GPU


# 数据预处理
def preprocess_data(data, labels, max_length=64):
    input_ids, attention_masks, processed_labels = [], [], []

    for i, text in enumerate(data):
        text = " ".join([word for word in jieba.cut(text) if word not in stopwords])
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        processed_labels.append(labels[i])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    processed_labels = torch.tensor(processed_labels)

    return input_ids, attention_masks, processed_labels

train_inputs, train_masks, train_labels = preprocess_data(train_data, train_labels)
test_inputs, test_masks, test_labels = preprocess_data(test_data, test_labels)

# 创建DataLoader
batch_size = 16
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
validation_dataset = TensorDataset(test_inputs, test_masks, test_labels)
validation_dataloader = DataLoader(validation_dataset, sampler=SequentialSampler(validation_dataset), batch_size=batch_size)

# 设置优化器和调度器
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 8
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 训练模型
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

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
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))


# #保存模型
# otpt_dir= "model/"
# model_to_save = model.module if hasattr(model, 'module') else model  # 处理多GPU情况
# model_to_save.save_pretrained(otpt_dir)
# tokenizer.save_pretrained(otpt_dir)
# 评估模型
print("\nRunning Validation...")
t0 = time.time()
model.eval()
predictions, true_labels = [], []



for batch in validation_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    logits = outputs[0]
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
output_file_path = "./output/output_comparison.txt"
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write("True Label\tPredicted Label\n")
    for true_label, predicted_label in zip(flat_true_labels,flat_predictions):
        true_label_str = '\t'.join(str(true_label))
        predicted_label_str = '\t'.join(str(predicted_label))
        output_file.write(f"{true_label_str}\t{predicted_label_str}\n")