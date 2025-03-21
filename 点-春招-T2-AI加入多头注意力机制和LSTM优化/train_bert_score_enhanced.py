import torch
import numpy as np
import pandas as pd
import json
import logging
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, precision_recall_fscore_support
import ast
import re
from tqdm import tqdm
import pickle
from torch.nn import functional as F

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置随机种子
def set_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed(42)

# 文本清洗函数
def clean_text(text):
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s，。！？,.!?、；：""''（）【】《》]', '', text)
    text = re.sub(r'([,.!?，。！？、；：])\1+', r'\1', text)
    return text.strip()

# 简单的数据增强函数
def augment_text(text, score):
    augmented_samples = []
    
    # 原始样本也保留
    augmented_samples.append((text, score))
    
    # 如果分数接近决策边界(5-7分)，生成更多样本
    if 5 <= score <= 7:
        # 随机删除一些字符
        if len(text) > 10:
            n_chars_to_delete = min(5, len(text) // 20)
            for _ in range(2):  # 生成2个这样的样本
                aug_text = text
                for _ in range(n_chars_to_delete):
                    if len(aug_text) <= 1:
                        break
                    pos = np.random.randint(0, len(aug_text) - 1)
                    aug_text = aug_text[:pos] + aug_text[pos+1:]
                augmented_samples.append((aug_text, score))
        
        # 随机替换一些常见标点符号
        punct_mapping = {'，': ',', '。': '.', '！': '!', '？': '?', '；': ';', '：': ':'}
        aug_text = text
        for k, v in punct_mapping.items():
            if k in aug_text and np.random.random() < 0.3:
                aug_text = aug_text.replace(k, v)
        if aug_text != text:
            augmented_samples.append((aug_text, score))
    
    return augmented_samples

# 自定义数据集
class MovieReviewDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_length=128):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        score = float(self.scores[idx])
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 二分类标签 (0: 负面, 1: 正面)
        binary_label = 1 if score >= 6 else 0
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'score': torch.tensor(score, dtype=torch.float),
            'binary_label': torch.tensor(binary_label, dtype=torch.long)
        }

# 增强的BERT模型 - 多任务学习（同时进行回归和分类）
class BertMultiTaskModel(nn.Module):
    def __init__(self, bert_model_name, dropout_rate=0.3):
        super(BertMultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # 冻结BERT底层参数以避免过拟合
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < 8:  # 冻结前8层
                for param in layer.parameters():
                    param.requires_grad = False
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.bert.config.hidden_size,
            num_heads=12,
            dropout=dropout_rate
        )
        
        # 特征提取层
        hidden_size = self.bert.config.hidden_size
        
        # 双向LSTM增强特征提取
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        # Dropout和全连接层
        self.dropout = nn.Dropout(dropout_rate)
        self.dense1 = nn.Linear(hidden_size * 2, 512)  # *2因为有池化输出和注意力输出
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.activation1 = nn.LeakyReLU(0.1)
        
        self.dense2 = nn.Linear(512, 256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.activation2 = nn.LeakyReLU(0.1)
        
        # 回归任务头（分数预测）
        self.regressor = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )
        
        # 分类任务头（积极/消极二分类）
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 2)  # 二分类
        )
        
        # 残差连接
        self.residual1 = nn.Linear(hidden_size * 2, 512)
        self.residual2 = nn.Linear(512, 256)
    
    def forward(self, input_ids, attention_mask=None):
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 获取序列输出和池化输出
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # BiLSTM处理序列
        lstm_output, _ = self.bilstm(sequence_output)
        
        # 多头注意力
        attn_output, _ = self.multihead_attn(
            sequence_output.permute(1, 0, 2),
            sequence_output.permute(1, 0, 2),
            sequence_output.permute(1, 0, 2)
        )
        attn_output = attn_output.permute(1, 0, 2).mean(dim=1)
        
        # 合并所有特征
        combined_output = torch.cat([attn_output, pooled_output], dim=1)
        
        # 前向传播（带残差连接）
        x = self.dropout(combined_output)
        residual = self.residual1(x)
        x = self.dense1(x)
        x = self.batch_norm1(x)
        x = self.activation1(x + residual)
        
        residual = self.residual2(x)
        x = self.dense2(x)
        x = self.batch_norm2(x)
        x = self.activation2(x + residual)
        
        # 分数预测（回归任务）
        score = self.regressor(x)
        score = torch.sigmoid(score) * 9 + 1  # 映射到1-10范围
        
        # 情感分类（分类任务）
        logits = self.classifier(x)
        
        return {
            'score': score.squeeze(),
            'logits': logits
        }

# 结合MSE和交叉熵的多任务损失函数
class MultiTaskLoss(nn.Module):
    def __init__(self, regression_weight=0.4, classification_weight=0.6):
        super(MultiTaskLoss, self).__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, outputs, score_targets, class_targets):
        reg_loss = self.mse_loss(outputs['score'], score_targets)
        cls_loss = self.ce_loss(outputs['logits'], class_targets)
        
        # 总损失是加权和
        total_loss = (self.regression_weight * reg_loss) + (self.classification_weight * cls_loss)
        return total_loss, reg_loss, cls_loss

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=15):
    best_val_loss = float('inf')
    best_val_f1 = 0
    patience = 5
    patience_counter = 0
    
    # 训练历史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_reg_loss': [], 'val_reg_loss': [],
        'train_cls_loss': [], 'val_cls_loss': [],
        'val_f1': []
    }
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_reg_loss = 0
        train_cls_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)
            labels = batch['binary_label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            
            # 计算多任务损失
            loss, reg_loss, cls_loss = criterion(outputs, scores, labels)
            
            # L2正则化
            l2_reg = torch.tensor(0., requires_grad=True, device=device)
            for param in model.parameters():
                l2_reg = l2_reg + torch.norm(param, 2)
            
            loss = loss + 0.001 * l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_reg_loss += reg_loss.item()
            train_cls_loss += cls_loss.item()
            train_steps += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': '{:.3f}'.format(train_loss / train_steps),
                'reg_loss': '{:.3f}'.format(train_reg_loss / train_steps),
                'cls_loss': '{:.3f}'.format(train_cls_loss / train_steps)
            })
        
        # 保存训练历史
        history['train_loss'].append(train_loss / train_steps)
        history['train_reg_loss'].append(train_reg_loss / train_steps)
        history['train_cls_loss'].append(train_cls_loss / train_steps)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_reg_loss = 0
        val_cls_loss = 0
        val_steps = 0
        all_preds = []
        all_labels = []
        all_scores = []
        all_true_scores = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                scores = batch['score'].to(device)
                labels = batch['binary_label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                
                # 计算损失
                loss, reg_loss, cls_loss = criterion(outputs, scores, labels)
                val_loss += loss.item()
                val_reg_loss += reg_loss.item()
                val_cls_loss += cls_loss.item()
                val_steps += 1
                
                # 获取分类预测结果
                preds = torch.argmax(outputs['logits'], dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
                # 收集分数预测结果
                all_scores.extend(outputs['score'].cpu().tolist())
                all_true_scores.extend(scores.cpu().tolist())
        
        val_loss /= val_steps
        val_reg_loss /= val_steps
        val_cls_loss /= val_steps
        
        # 计算F1分数
        precision, recall, val_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        # 计算MSE（分数预测）
        val_mse = mean_squared_error(all_true_scores, all_scores)
        val_mae = mean_absolute_error(all_true_scores, all_scores)
        
        # 保存验证历史
        history['val_loss'].append(val_loss)
        history['val_reg_loss'].append(val_reg_loss)
        history['val_cls_loss'].append(val_cls_loss)
        history['val_f1'].append(val_f1)
        
        # 打印结果
        logging.info(f'Epoch {epoch + 1}:')
        logging.info(f'Training Loss: {train_loss / train_steps:.4f} (Reg: {train_reg_loss / train_steps:.4f}, Cls: {train_cls_loss / train_steps:.4f})')
        logging.info(f'Validation Loss: {val_loss:.4f} (Reg: {val_reg_loss:.4f}, Cls: {val_cls_loss:.4f})')
        logging.info(f'Regression - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}')
        logging.info(f'Classification - F1: {val_f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
        
        # 保存最佳模型 - 以F1分数为标准
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存模型
            model_path = './model_score_enhanced'
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            
            torch.save(model.state_dict(), f'{model_path}/model.pt')
            model.bert.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            # 保存配置和历史
            config = {
                'dropout_rate': 0.3,
                'best_val_loss': best_val_loss,
                'best_val_f1': best_val_f1,
                'epoch': epoch + 1
            }
            with open(f'{model_path}/config.json', 'w') as f:
                json.dump(config, f)
            
            with open(f'{model_path}/training_history.pkl', 'wb') as f:
                pickle.dump(history, f)
            
            logging.info(f'New best model saved! F1: {best_val_f1:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after epoch {epoch + 1}')
                break
    
    return best_val_loss, best_val_f1, history

# 主函数
if __name__ == "__main__":
    # 加载数据
    logging.info("加载数据...")
    with open('./url_list.txt', 'r', encoding='utf-8') as file:
        data = file.read()
    
    data_cleaned = data.replace('\n', ' ')
    data_list = ast.literal_eval(data_cleaned)
    
    # 数据预处理和增强
    logging.info("数据预处理和增强...")
    texts = []
    scores = []
    
    for item in data_list:
        text = clean_text(item[0])
        score = item[1]
        
        # 数据增强
        augmented_samples = augment_text(text, score)
        for aug_text, aug_score in augmented_samples:
            texts.append(aug_text)
            scores.append(aug_score)
    
    logging.info(f"原始样本数: {len(data_list)}, 增强后样本数: {len(texts)}")
    
    # 统计类别分布
    pos_count = sum(1 for s in scores if s >= 6)
    neg_count = len(scores) - pos_count
    logging.info(f"类别分布 - 正面: {pos_count}, 负面: {neg_count}")
    
    # 划分训练集和验证集
    train_texts, val_texts, train_scores, val_scores = train_test_split(
        texts, scores, test_size=0.15, random_state=42, stratify=[1 if s >= 6 else 0 for s in scores]
    )
    
    logging.info(f"训练集: {len(train_texts)}样本, 验证集: {len(val_texts)}样本")
    
    # 加载分词器和模型
    logging.info("初始化模型和分词器...")
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    model = BertMultiTaskModel('./bert-base-chinese')
    
    # 创建数据加载器
    train_dataset = MovieReviewDataset(train_texts, train_scores, tokenizer)
    val_dataset = MovieReviewDataset(val_texts, val_scores, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=24)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 损失函数
    criterion = MultiTaskLoss(regression_weight=0.4, classification_weight=0.6)
    
    # 优化器 - 分层学习率
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': 2e-5,
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': 2e-5,
            'weight_decay': 0.0
        },
        {
            'params': [p for n, p in model.named_parameters() if not n.startswith('bert.')],
            'lr': 3e-4,
            'weight_decay': 0.01
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters)
    
    # 学习率调度器
    total_steps = len(train_loader) * 15
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # 训练模型
    logging.info("开始训练...")
    best_val_loss, best_val_f1, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=15
    )
    
    logging.info(f"训练完成！最佳验证损失: {best_val_loss:.4f}, 最佳F1分数: {best_val_f1:.4f}") 