import torch
import numpy as np
import json
import logging
import os
import re
import ast
import random
from torch import nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

#多头看了3后才知道可以这么优化，这段代码是AI生成的
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
    text = re.sub(r'http\S+', '', text)
    text = ' '.join(text.split())
    return text

# 从url_list.txt读取测试样本
def load_test_data(max_samples=50):
    test_texts = []
    test_labels = []
    original_scores = []
    
    try:
        # 读取数据
        with open('./url_list.txt', 'r', encoding='utf-8') as file:
            data = file.read()
        
        # 数据清洗，去除换行符
        data_cleaned = data.replace('\n', ' ')
        data_list = ast.literal_eval(data_cleaned)
        
        # 随机选择样本
        if len(data_list) > max_samples:
            test_samples = random.sample(data_list, max_samples)
        else:
            test_samples = data_list
        
        for sample in test_samples:
            text = sample[0]
            original_score = sample[1]
            
            # 清洗文本
            text = clean_text(text)
            
            test_texts.append(text)
            original_scores.append(original_score)
            # 将原始评分转换为情感类别 (<=5分为负面，>5分为正面)
            test_labels.append(0 if original_score <= 5 else 1)
        
        print(f"成功加载{len(test_texts)}个测试样本")
        return test_texts, test_labels, original_scores
    except Exception as e:
        print(f"从文件加载数据时出错: {e}")
        return [], [], []

# 主函数
if __name__ == "__main__":
    # 加载模型
    model_path = './model_score_enhanced'  # 增强版模型路径
    
    try:
        # 加载分词器
        logging.info("加载分词器...")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # 加载模型
        logging.info("加载增强版模型...")
        model = BertMultiTaskModel('./bert-base-chinese')
        model.load_state_dict(torch.load(f'{model_path}/model.pt', map_location=torch.device('cpu')), strict=False)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        logging.info(f"模型加载成功，使用设备: {device}")
    
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        try:
            # 回退到二分类模型
            logging.info("尝试加载标准二分类模型...")
            from transformers import AutoModelForSequenceClassification
            model_path = './model'  # 标准模型路径
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = BertTokenizer.from_pretrained(model_path)
            model.to(device)
            model.eval()
            logging.info(f"标准模型加载成功，使用设备: {device}")
        except:
            logging.error("所有模型加载失败，程序退出")
            exit(1)
    
    # 加载测试数据
    test_texts, test_labels, original_scores = load_test_data(50)

    # 打印样本统计信息
    positive_count = sum(1 for label in test_labels if label == 1)
    negative_count = len(test_labels) - positive_count
    logging.info(f"正面评论数量: {positive_count}")
    logging.info(f"负面评论数量: {negative_count}")

    # 预测
    logging.info("开始预测...")
    batch_size = 8
    predictions = []
    binary_predictions = []
    confidence_scores = []
    
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i:i+batch_size]
            
            # 编码文本
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            
            # 移除token_type_ids参数
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
                
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 预测
            if isinstance(model, BertMultiTaskModel):
                # 使用增强版模型
                outputs = model(**inputs)
                
                # 收集分数预测
                scores = outputs['score'].cpu().numpy()
                predictions.extend(scores.tolist())
                
                # 收集二分类预测
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                binary_predictions.extend(preds.tolist())
                
                # 计算置信度
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                confidence_scores.extend(probs.tolist())
            else:
                # 使用标准二分类模型
                outputs = model(**inputs)
                logits = outputs.logits
                
                # 获取预测结果
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                binary_predictions.extend(preds.tolist())
                
                # 计算分数估计（基于概率）
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                confidence_scores.extend(probs.tolist())
                
                # 根据概率估计分数
                estimated_scores = []
                for prob in probs:
                    # 将概率映射到1-10分
                    # 负面概率高 → 1-5分，正面概率高 → 6-10分
                    score = 1 + 4 * prob[0] if prob[0] > 0.5 else 6 + 4 * prob[1]
                    estimated_scores.append(score)
                predictions.extend(estimated_scores)

    # 计算二分类指标
    accuracy = accuracy_score(test_labels, binary_predictions)
    precision = precision_score(test_labels, binary_predictions, average='binary', zero_division=0)
    recall = recall_score(test_labels, binary_predictions, average='binary', zero_division=0)
    f1 = f1_score(test_labels, binary_predictions, average='binary', zero_division=0)

    # 计算分数预测误差
    mse = np.mean((np.array(predictions) - np.array(original_scores))**2)
    mae = np.mean(np.abs(np.array(predictions) - np.array(original_scores)))
    rmse = np.sqrt(mse)
    
    # 计算±0.5分和±1分内的准确率
    accuracy_0_5 = np.mean(np.abs(np.array(predictions) - np.array(original_scores)) <= 0.5)
    accuracy_1 = np.mean(np.abs(np.array(predictions) - np.array(original_scores)) <= 1.0)

    # 输出结果
    logging.info("\n" + "=" * 50)
    logging.info(f"测试样本数量: {len(test_texts)}")
    logging.info(f"二分类指标:")
    logging.info(f"  准确率 (Accuracy): {accuracy:.4f}")
    logging.info(f"  精确率 (Precision): {precision:.4f}")
    logging.info(f"  召回率 (Recall): {recall:.4f}")
    logging.info(f"  F1分数: {f1:.4f}")
    logging.info("-" * 50)
    logging.info(f"分数预测指标:")
    logging.info(f"  均方误差 (MSE): {mse:.4f}")
    logging.info(f"  均方根误差 (RMSE): {rmse:.4f}")
    logging.info(f"  平均绝对误差 (MAE): {mae:.4f}")
    logging.info(f"  ±0.5分准确率: {accuracy_0_5:.4f}")
    logging.info(f"  ±1分准确率: {accuracy_1:.4f}")
    logging.info("=" * 50)

    # 输出详细的分类报告
    logging.info("\n分类报告:")
    logging.info(classification_report(test_labels, binary_predictions, 
                          target_names=['负面 (1-5分)', '正面 (6-10分)'],
                          zero_division=0))

    # 收集所有预测结果
    results = []
    for i, (text, true_score, pred_score, binary_pred, true_label) in enumerate(
        zip(test_texts, original_scores, predictions, binary_predictions, test_labels)
    ):
        sentiment_correct = binary_pred == true_label
        
        results.append({
            "id": i + 1,
            "text": text[:100] + "..." if len(text) > 100 else text,
            "true_score": float(true_score),
            "predicted_score": float(pred_score),
            "score_error": float(abs(true_score - pred_score)),
            "true_sentiment": "正面" if true_label == 1 else "负面",
            "predicted_sentiment": "正面" if binary_pred == 1 else "负面",
            "sentiment_correct": bool(sentiment_correct)
        })

    # 输出一些示例预测
    logging.info("\n预测示例:")
    for i in range(min(5, len(results))):
        sample = results[i]
        logging.info(f"{i+1}. 文本: {sample['text']}")
        logging.info(f"   真实评分: {sample['true_score']}, 预测评分: {sample['predicted_score']:.1f}, 误差: {sample['score_error']:.2f}")
        logging.info(f"   真实情感: {sample['true_sentiment']}, 预测情感: {sample['predicted_sentiment']}, 正确: {sample['sentiment_correct']}")
        logging.info("-" * 40)

    # 找出预测错误的样本
    errors = [item for item in results if not item["sentiment_correct"]]
    logging.info(f"\n错误预测数量: {len(errors)}")

    if errors:
        logging.info("\n错误预测示例:")
        for i, error in enumerate(errors[:3]):  # 只显示前3个错误
            logging.info(f"{i+1}. 文本: {error['text']}")
            logging.info(f"   原始评分: {error['true_score']}, 真实情感: {error['true_sentiment']}")
            logging.info(f"   预测评分: {error['predicted_score']:.1f}, 预测情感: {error['predicted_sentiment']}")
            logging.info("-" * 40)

    # 将预测结果保存到文件
    with open("yangzheng_50_enhanced_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.info("\n预测结果已保存到 yangzheng_50_enhanced_results.json") 