import torch
import time
import pandas as pd
import numpy as np
import os
import ast
import joblib
import random
import re
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from torch.amp import autocast, GradScaler
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed(42)

# 配置日志记录
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 创建数据目录
os.makedirs('./data', exist_ok=True)
os.makedirs('./model', exist_ok=True)

# 简单的数据增强函数
def augment_text(text):
    # 1. 随机删除一些字符
    if len(text) > 10 and random.random() < 0.5:
        n_chars_to_delete = random.randint(1, min(5, len(text) // 10))
        for _ in range(n_chars_to_delete):
            pos = random.randint(0, len(text) - 1)
            text = text[:pos] + text[pos+1:]
    
    # 2. 随机替换一些常见标点符号
    if random.random() < 0.3:
        text = text.replace('，', ',').replace('。', '.').replace('！', '!').replace('？', '?')
    
    # 3. 随机调换相邻字符位置
    if len(text) > 5 and random.random() < 0.3:
        pos = random.randint(0, len(text) - 2)
        text = text[:pos] + text[pos+1] + text[pos] + text[pos+2:]
    return text

# 读取数据
try:
    with open('./url_list.txt', 'r', encoding='utf-8') as file:
        data = file.read()

    # 数据清洗，去除换行符
    data_cleaned = data.replace('\n', ' ')
    data_list = ast.literal_eval(data_cleaned)
    data = pd.DataFrame(data_list, columns=['Text', 'Value'])
    
    # 去除重复数据
    data = data.drop_duplicates(subset='Text')
    logging.info(f"去重后数据量: {len(data)}")
    
    # 去除过短的评论
    data = data[data['Text'].str.len() > 5]
    logging.info(f"过滤短评论后数据量: {len(data)}")
    
    # 清洗文本
    def clean_text(text):
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 去除URL
        text = re.sub(r'http\S+', '', text)
        # 去除特殊字符
        text = re.sub(r'[^\w\s，。！？,.!?]', '', text)
        # 去除重复的标点符号
        text = re.sub(r'[,.!?，。！？]{2,}', lambda m: m.group()[0], text)
        return text.strip()
    
    data['Text'] = data['Text'].apply(clean_text)
    
    # 将评分转换为2类：负面(1-5)、正面(6-10)
    def map_score_to_sentiment(score):
        if score <= 5:
            return 0  # 负面
        else:
            return 1  # 正面

    data['Sentiment'] = data['Value'].apply(map_score_to_sentiment)
    logging.info(f"情感分布: 负面={len(data[data['Sentiment']==0])}, 正面={len(data[data['Sentiment']==1])}")
    
    # 数据增强 - 为每个类别添加更多样本
    augmented_data = []
    for sentiment in range(2):
        subset = data[data['Sentiment'] == sentiment]
        if len(subset) < 500:  # 增加目标样本数
            # 获取原始数据的文本和情感标签
            texts = subset['Text'].tolist()
            sentiments = subset['Sentiment'].tolist()
            
            # 对每个样本生成3个增强版本
            for i in range(len(texts)):
                for _ in range(3):  # 增加每个样本的增强次数
                    augmented_text = augment_text(texts[i])
                    augmented_data.append((augmented_text, sentiments[i]))
    
    # 将增强数据添加到原始数据中
    augmented_df = pd.DataFrame(augmented_data, columns=['Text', 'Sentiment'])
    data = pd.concat([data, augmented_df], ignore_index=True)
    
    logging.info(f"数据增强后数据量: {len(data)}")
    
    # 数据均衡处理
    balanced_data = pd.DataFrame()
    for sentiment in range(2):
        subset = data[data['Sentiment'] == sentiment]
        # 最多取400个样本
        if len(subset) > 400:
            subset = subset.sample(400, random_state=42)
        balanced_data = pd.concat([balanced_data, subset])
    
    data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱数据
    logging.info(f"均衡后最终数据量: {len(data)}")
    
except Exception as e:
    logging.error(f"Error processing data: {e}")
    raise

# 最优模型训练结果的保存路径
best_model_path = './model'
X = data['Text']  # 特征列
y = data['Sentiment'].values  # 使用情感分类作为标签

# 对标签数据进行编码转换
logging.info("1、开始编码转换")
label_encoder = LabelEncoder()  # 初始化
y_encoded = label_encoder.fit_transform(y)
logging.info(f'分类数：{len(label_encoder.classes_)}')
logging.info(f'标签分布：{np.bincount(y_encoded)}')

# 保存 label_encoder 以便以后使用
joblib.dump(label_encoder, './data/encoder.joblib')

# 分割数据集 - 使用交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(cv.split(X, y_encoded))
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

logging.info(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")

# 加载BERT分词器
try:
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    tokenizer.save_pretrained(best_model_path)
except:
    logging.info("尝试从Hugging Face下载模型")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokenizer.save_pretrained(best_model_path)

# BERT预处理 - 将文本数据转换成BERT模型能够理解的格式
def preprocess_for_bert(data, labels, max_length=128):
    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids.append(encoded_sent['input_ids'])
        attention_masks.append(encoded_sent['attention_mask'])

    # 转换为PyTorch张量
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels

# 预处理数据
logging.info("2、开始预处理数据")
train_inputs, train_masks, train_labels = preprocess_for_bert(X_train, y_train, max_length=128)
val_inputs, val_masks, val_labels = preprocess_for_bert(X_val, y_val, max_length=128)

# 创建DataLoader
batch_size = 32  # 增加批处理大小
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(val_inputs, val_masks, val_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# 加载BERT模型
logging.info("3、开始加载模型")
try:
    model = BertForSequenceClassification.from_pretrained('./bert-base-chinese', 
                                                        num_labels=len(label_encoder.classes_),
                                                        ignore_mismatched_sizes=True)
except:
    logging.info("尝试从Hugging Face下载模型")
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', 
                                                        num_labels=len(label_encoder.classes_),
                                                        ignore_mismatched_sizes=True)

# 检测GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 在bert最后一层添加一个简单的dropout层
model.bert.encoder.layer[-1].output.dropout.p = 0.2

# 设置优化器和学习率调度器
EPOCHS = 10
# 使用学习率衰减和特定的optimizer设置
optimizer = AdamW([
    {'params': model.bert.parameters(), 'lr': 5e-6, 'weight_decay': 0.01},  # BERT层较低学习率
    {'params': model.classifier.parameters(), 'lr': 5e-4, 'weight_decay': 0.1}  # 分类层较高学习率
])
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                         num_warmup_steps=int(total_steps*0.1),
                                         num_training_steps=total_steps)

# 计算准确度
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# 训练过程中使用混合精度训练
scaler = GradScaler()

# 训练和评估
logging.info("4、开始训练")
best_val_accuracy = 0
for epoch in range(EPOCHS):
    logging.info(f'Epoch {epoch + 1}/{EPOCHS}')
    
    # 训练阶段
    model.train()
    total_train_loss = 0
    total_train_accuracy = 0
    
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device).long()

        model.zero_grad()
        
        # 使用混合精度训练
        if torch.cuda.is_available():
            with autocast(device_type='cuda'):
                outputs = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(b_input_ids, 
                         token_type_ids=None, 
                         attention_mask=b_input_mask, 
                         labels=b_labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        total_train_loss += loss.item()
        
        # 计算训练准确率
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_train_accuracy += flat_accuracy(logits, label_ids)
        
        # 每10步打印一次进度
        if (step % 10 == 0 and step != 0) or (step == len(train_dataloader) - 1):
            logging.info(f'  Batch {step}/{len(train_dataloader)} - Loss: {loss.item():.4f}')
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    
    logging.info(f'  Average training loss: {avg_train_loss:.4f}')
    logging.info(f'  Average training accuracy: {avg_train_accuracy:.4f}')
    
    # 释放内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 验证阶段
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device).long()
        
        with torch.no_grad():
            outputs = model(b_input_ids, 
                          token_type_ids=None, 
                          attention_mask=b_input_mask, 
                          labels=b_labels)
        
        loss = outputs.loss
        total_eval_loss += loss.item()
        
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        preds = np.argmax(logits, axis=1).flatten()
        all_preds.extend(preds)
        all_labels.extend(label_ids.flatten())
        
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # 计算验证集准确率
    validation_accuracy = accuracy_score(all_labels, all_preds)
    
    logging.info(f'  Validation Loss: {avg_val_loss:.4f}')
    logging.info(f'  Validation Accuracy: {avg_val_accuracy:.4f}')
    logging.info(f'  sklearn Validation Accuracy: {validation_accuracy:.4f}')
    
    # 保存最佳模型
    if validation_accuracy > best_val_accuracy:
        best_val_accuracy = validation_accuracy
        model.save_pretrained(best_model_path)
        logging.info(f'  Saved new best model with accuracy: {best_val_accuracy:.4f}')
    
    # 释放内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logging.info("-------------------")

logging.info(f"训练完成！最佳验证准确率: {best_val_accuracy:.4f}") 