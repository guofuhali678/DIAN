import torch
import numpy as np
import json
import logging
import os
import re
import ast
import pickle
from torch import nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, precision_recall_fscore_support,
    classification_report, confusion_matrix, accuracy_score, f1_score
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载增强版模型的结构定义
class BertMultiTaskModel(nn.Module):
    def __init__(self, bert_model_name, dropout_rate=0.3):
        super(BertMultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, ignore_mismatched_sizes=True)
        
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

# 文本清洗函数
def clean_text(text):
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s，。！？,.!?、；：""''（）【】《》]', '', text)
    text = re.sub(r'([,.!?，。！？、；：])\1+', r'\1', text)
    return text.strip()

# 主函数
if __name__ == "__main__":
    model_path = './model_score_enhanced'
    test_samples = 100  # 设置测试样本数量

    # 加载模型
    logging.info("加载模型...")
    try:
        # 加载分词器
        tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # 初始化模型 - 先使用bert-base-chinese作为基础模型
        model = BertMultiTaskModel('./bert-base-chinese')
        
        # 加载模型权重
        model.load_state_dict(torch.load(f'{model_path}/model.pt', map_location=torch.device('cpu')), strict=False)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        logging.info(f"模型加载成功，使用设备: {device}")
        
        # 尝试加载训练历史
        try:
            with open(f'{model_path}/training_history.pkl', 'rb') as f:
                history = pickle.load(f)
            best_f1 = max(history['val_f1'])
            logging.info(f"从训练历史中获取的最佳F1分数: {best_f1:.4f}")
        except:
            logging.info("未找到训练历史文件")
    
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        raise

    # 加载测试数据
    logging.info("加载测试数据...")
    try:
        with open('./url_list.txt', 'r', encoding='utf-8') as file:
            data = file.read()
        
        data_cleaned = data.replace('\n', ' ')
        data_list = ast.literal_eval(data_cleaned)
        
        # 随机选择测试样本
        np.random.seed(42)
        test_indices = np.random.choice(len(data_list), min(test_samples, len(data_list)), replace=False)
        test_data = [data_list[i] for i in test_indices]
        
        test_texts = []
        original_scores = []
        for item in test_data:
            text = clean_text(item[0])
            score = item[1]
            test_texts.append(text)
            original_scores.append(score)
        
        logging.info(f"成功加载{len(test_texts)}个测试样本")
    except Exception as e:
        logging.error(f"数据加载失败: {e}")
        raise

    # 预测
    logging.info("开始预测...")
    predictions = []
    binary_predictions = []
    confidence_scores = []

    batch_size = 16
    num_batches = (len(test_texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(test_texts))
            
            batch_texts = test_texts[start_idx:end_idx]
            
            # 编码
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # 移至设备
            input_ids = encoded_input['input_ids'].to(device)
            attention_mask = encoded_input['attention_mask'].to(device)
            
            # 预测
            outputs = model(input_ids, attention_mask)
            
            # 收集分数
            pred_scores = outputs['score'].cpu().numpy()
            predictions.extend(pred_scores.tolist())
            
            # 收集二分类结果
            logits = outputs['logits']
            pred_classes = torch.argmax(logits, dim=1).cpu().numpy()
            binary_predictions.extend(pred_classes.tolist())
            
            # 计算置信度
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            confidence_scores.extend(probs.tolist())

    # 将原始评分转换为二分类标签
    binary_labels = [1 if score >= 6 else 0 for score in original_scores]

    # 计算回归指标
    mse = mean_squared_error(original_scores, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(original_scores, predictions)

    # 计算分类指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_labels, binary_predictions, average='binary', zero_division=0
    )
    accuracy = accuracy_score(binary_labels, binary_predictions)

    # 计算在±0.5分和±1分内的准确率
    accuracy_0_5 = np.mean(np.abs(np.array(predictions) - np.array(original_scores)) <= 0.5)
    accuracy_1 = np.mean(np.abs(np.array(predictions) - np.array(original_scores)) <= 1.0)

    # 输出结果
    logging.info("=" * 50)
    logging.info(f"测试集大小: {len(test_texts)}个样本")
    logging.info(f"均方误差 (MSE): {mse:.4f}")
    logging.info(f"均方根误差 (RMSE): {rmse:.4f}")
    logging.info(f"平均绝对误差 (MAE): {mae:.4f}")
    logging.info(f"±0.5分准确率: {accuracy_0_5:.4f}")
    logging.info(f"±1分准确率: {accuracy_1:.4f}")
    logging.info("-" * 50)
    logging.info("二分类指标 (分数≥6为正面，<6为负面):")
    logging.info(f"准确率 (Accuracy): {accuracy:.4f}")
    logging.info(f"精确率 (Precision): {precision:.4f}")
    logging.info(f"召回率 (Recall): {recall:.4f}")
    logging.info(f"F1分数: {f1:.4f}")
    logging.info("=" * 50)

    # 输出详细分类报告
    logging.info("分类报告:")
    logging.info(classification_report(
        binary_labels, binary_predictions,
        target_names=['负面 (<6分)', '正面 (≥6分)'],
        zero_division=0
    ))

    # 收集所有预测结果
    results = []
    for i, (text, true_score, pred_score, binary_pred, conf_score) in enumerate(
        zip(test_texts, original_scores, predictions, binary_predictions, confidence_scores)
    ):
        binary_label = 1 if true_score >= 6 else 0
        sentiment_correct = binary_pred == binary_label
        
        results.append({
            "id": i + 1,
            "text": text[:100] + "..." if len(text) > 100 else text,
            "true_score": float(true_score),
            "predicted_score": float(pred_score),
            "score_error": float(abs(true_score - pred_score)),
            "true_sentiment": "正面" if binary_label == 1 else "负面",
            "predicted_sentiment": "正面" if binary_pred == 1 else "负面",
            "sentiment_correct": bool(sentiment_correct),
            "negative_confidence": float(conf_score[0]),
            "positive_confidence": float(conf_score[1])
        })

    # 找出预测最准确和最不准确的5个样本
    results_sorted_by_error = sorted(results, key=lambda x: x["score_error"])
    most_accurate = results_sorted_by_error[:5]
    least_accurate = results_sorted_by_error[-5:]

    logging.info("\n最准确的5个预测:")
    for i, sample in enumerate(most_accurate):
        logging.info(f"{i+1}. 文本: {sample['text']}")
        logging.info(f"   真实评分: {sample['true_score']}, 预测评分: {sample['predicted_score']:.1f}, 误差: {sample['score_error']:.2f}")
        logging.info(f"   真实情感: {sample['true_sentiment']}, 预测情感: {sample['predicted_sentiment']}, 正确: {sample['sentiment_correct']}")
        logging.info("-" * 30)

    logging.info("\n最不准确的5个预测:")
    for i, sample in enumerate(least_accurate):
        logging.info(f"{i+1}. 文本: {sample['text']}")
        logging.info(f"   真实评分: {sample['true_score']}, 预测评分: {sample['predicted_score']:.1f}, 误差: {sample['score_error']:.2f}")
        logging.info(f"   真实情感: {sample['true_sentiment']}, 预测情感: {sample['predicted_sentiment']}, 正确: {sample['sentiment_correct']}")
        logging.info("-" * 30)

    # 保存预测结果
    with open("enhanced_model_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.info(f"\n预测结果已保存到 enhanced_model_results.json") 