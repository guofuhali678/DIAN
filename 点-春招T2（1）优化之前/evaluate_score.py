import torch
import numpy as np
import pandas as pd
import re
import ast
import json
import logging
from transformers import BertTokenizer, BertModel
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score, classification_report

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置随机种子
def set_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed(42)

# 自定义BERT回归模型（需要与训练时相同）
class BertRegressionModel(nn.Module):
    def __init__(self, bert_model_name, dropout_rate=0.3):
        super(BertRegressionModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 回归头 - 使用多层以增强复杂度
        self.dense1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.dense2 = nn.Linear(512, 256)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.dense3 = nn.Linear(256, 64)
        self.activation3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # 最后一层，输出一个分数（回归）
        self.regressor = nn.Linear(64, 1)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # BERT前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取[CLS]标记的输出
        pooled_output = outputs.pooler_output
        
        # 回归头
        x = self.dropout(pooled_output)
        x = self.dense1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        
        x = self.dense2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        
        x = self.dense3(x)
        x = self.activation3(x)
        x = self.dropout3(x)
        
        # 输出分数并限制在1-10范围内
        score = self.regressor(x)
        score = torch.sigmoid(score) * 9 + 1  # 将0-1映射到1-10
        
        return score

# 文本清洗函数
def clean_text(text):
    # 去除URL
    text = re.sub(r'http[s]?://\S+', '', text)
    # 标准化空格
    text = re.sub(r'\s+', ' ', text)
    # 去除特殊字符，保留中文标点
    text = re.sub(r'[^\w\s，。！？,.!?、；：""''（）【】《》]', '', text)
    # 合并重复标点
    text = re.sub(r'([,.!?，。！？、；：])\1+', r'\1', text)
    return text.strip()

# 加载模型和分词器
try:
    # 检查是否有训练好的回归模型
    model_path = './model_score'
    
    # 尝试加载配置
    with open(f'{model_path}/config.json', 'r') as f:
        config = json.load(f)
    
    # 初始化模型
    model = BertRegressionModel(
        './bert-base-chinese',
        dropout_rate=config.get('dropout_rate', 0.3)
    )
    
    # 加载预训练权重
    model.load_state_dict(torch.load(f'{model_path}/model.pt', map_location=torch.device('cpu')))
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # 设置为评估模式
    
    logging.info(f"模型和分词器加载成功，使用设备: {device}")
    
    # 模型类型
    regression_model = True
    
except Exception as e:
    logging.error(f"回归模型加载失败: {e}")
    logging.info("尝试加载分类模型...")
    
    try:
        # 如果回归模型不可用，尝试加载之前的分类模型
        model_path = './model'
        from transformers import BertForSequenceClassification
        
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        logging.info(f"分类模型加载成功，使用设备: {device}")
        
        # 模型类型
        regression_model = False
        
    except Exception as e2:
        logging.error(f"分类模型也加载失败: {e2}")
        raise Exception("无法加载任何可用模型")

# 加载测试数据
try:
    logging.info("加载测试数据...")
    with open('./url_list.txt', 'r', encoding='utf-8') as file:
        data = file.read()
    
    # 数据处理
    data_cleaned = data.replace('\n', ' ')
    data_list = ast.literal_eval(data_cleaned)
    
    # 随机选择100个样本作为测试集
    import random
    random.seed(42)
    test_samples = random.sample(data_list, min(100, len(data_list)))
    
    # 提取文本和分数
    test_texts = []
    original_scores = []
    
    for sample in test_samples:
        text = clean_text(sample[0])
        test_texts.append(text)
        original_scores.append(sample[1])
    
    logging.info(f"成功加载{len(test_texts)}个测试样本")
except Exception as e:
    logging.error(f"数据加载失败: {e}")
    raise

# 预测函数 - 回归模型
def predict_regression(texts, model, tokenizer, device, batch_size=16):
    all_predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 编码
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # 移动到GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            predictions = model(**inputs)
            # 将预测值限制在1-10范围内
            predictions = torch.clamp(predictions, 1.0, 10.0).cpu().numpy().flatten()
        
        all_predictions.extend(predictions)
    
    return all_predictions

# 预测函数 - 分类模型（将分类结果映射到评分）
def predict_classification(texts, model, tokenizer, device, batch_size=16):
    all_predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 编码
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # 移动到GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 获取每个类别的概率
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            # 对于二分类模型，我们根据积极情感的概率映射到评分
            # 如果P(positive)接近1，分数接近10；如果接近0，分数接近1
            if probs.shape[1] == 2:  # 二分类模型
                pos_probs = probs[:, 1]  # 积极情感的概率
                scores = pos_probs * 9 + 1  # 映射到1-10范围
            else:  # 多分类模型
                # 假设类别标签为1-10
                scores = np.zeros(len(batch_texts))
                for j in range(probs.shape[0]):
                    # 加权平均
                    scores[j] = np.sum(np.arange(1, probs.shape[1] + 1) * probs[j])
        
        all_predictions.extend(scores)
    
    return all_predictions

# 评估指标
def compute_metrics(predictions, targets):
    # 将预测值转换为numpy数组
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 确保预测值在1-10范围内
    predictions = np.clip(predictions, 1, 10)
    
    # 计算回归指标
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    
    # 计算准确率（预测值四舍五入后与真实值相等）
    rounded_preds = np.round(predictions)
    exact_accuracy = np.mean(rounded_preds == targets)
    
    # 计算在±0.5和±1分内的准确率
    accuracy_0_5 = np.mean(np.abs(predictions - targets) <= 0.5)
    accuracy_1 = np.mean(np.abs(predictions - targets) <= 1.0)
    
    # 计算分类指标 (将分数转换为二分类结果：<6为负面，>=6为正面)
    binary_preds = (predictions >= 6).astype(int)
    binary_targets = (targets >= 6).astype(int)
    
    precision = precision_score(binary_targets, binary_preds, zero_division=0)
    recall = recall_score(binary_targets, binary_preds, zero_division=0)
    f1 = f1_score(binary_targets, binary_preds, zero_division=0)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'exact_accuracy': exact_accuracy,
        'accuracy_0_5': accuracy_0_5,
        'accuracy_1': accuracy_1,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 进行预测
logging.info("开始预测...")
if regression_model:
    predictions = predict_regression(test_texts, model, tokenizer, device)
else:
    predictions = predict_classification(test_texts, model, tokenizer, device)

# 计算评估指标
metrics = compute_metrics(predictions, original_scores)

# 打印结果
logging.info("=" * 50)
logging.info(f"测试集大小: {len(test_texts)}个样本")
logging.info(f"均方误差 (MSE): {metrics['mse']:.4f}")
logging.info(f"均方根误差 (RMSE): {metrics['rmse']:.4f}")
logging.info(f"平均绝对误差 (MAE): {metrics['mae']:.4f}")
logging.info(f"精确匹配准确率: {metrics['exact_accuracy']:.4f}")
logging.info(f"±0.5分准确率: {metrics['accuracy_0_5']:.4f}")
logging.info(f"±1分准确率: {metrics['accuracy_1']:.4f}")
logging.info("-" * 50)
logging.info("二分类指标 (分数≥6为正面，<6为负面):")
logging.info(f"精确率 (Precision): {metrics['precision']:.4f}")
logging.info(f"召回率 (Recall): {metrics['recall']:.4f}")
logging.info(f"F1分数: {metrics['f1']:.4f}")
logging.info("=" * 50)

# 输出分类报告
binary_preds = (np.array(predictions) >= 6).astype(int)
binary_targets = (np.array(original_scores) >= 6).astype(int)
class_report = classification_report(binary_targets, binary_preds, target_names=['负面 (<6分)', '正面 (≥6分)'], zero_division=0)
logging.info(f"分类报告:\n{class_report}")

# 保存详细的预测结果
results = []
for i, (text, true_score, pred_score) in enumerate(zip(test_texts, original_scores, predictions)):
    # 计算二分类结果
    true_sentiment = 1 if true_score >= 6 else 0
    pred_sentiment = 1 if pred_score >= 6 else 0
    sentiment_correct = true_sentiment == pred_sentiment
    
    results.append({
        "id": i + 1,
        "text": text[:50] + "..." if len(text) > 50 else text,
        "true_score": float(true_score),
        "predicted_score": float(pred_score),
        "absolute_error": float(abs(true_score - pred_score)),
        "within_0.5": bool(abs(true_score - pred_score) <= 0.5),
        "within_1.0": bool(abs(true_score - pred_score) <= 1.0),
        "true_sentiment": int(true_sentiment),
        "predicted_sentiment": int(pred_sentiment),
        "sentiment_correct": bool(sentiment_correct)
    })

# 排序显示最准确和最不准确的预测
results_sorted = sorted(results, key=lambda x: x['absolute_error'])
logging.info("\n最准确的5个预测:")
for i, result in enumerate(results_sorted[:5]):
    logging.info(f"{i+1}. 文本: {result['text']}")
    logging.info(f"   真实评分: {result['true_score']:.1f}, 预测评分: {result['predicted_score']:.1f}, 误差: {result['absolute_error']:.2f}")
    logging.info("-" * 30)

logging.info("\n最不准确的5个预测:")
for i, result in enumerate(results_sorted[-5:]):
    logging.info(f"{i+1}. 文本: {result['text']}")
    logging.info(f"   真实评分: {result['true_score']:.1f}, 预测评分: {result['predicted_score']:.1f}, 误差: {result['absolute_error']:.2f}")
    logging.info("-" * 30)

# 保存结果到文件
with open("score_prediction_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

logging.info("评估完成，详细结果已保存到 score_prediction_results.json") 