import json
import random
import torch
import numpy as np
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 设置随机种子以确保结果可重现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 加载模型和分词器
model_path = './model'  # 假设模型保存的路径
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 将原始评分映射为情感类别
def map_score_to_sentiment(score):
    if score <= 5:
        return 0  # 负面
    else:
        return 1  # 正面

# 从url_list.txt中读取所有测试样本
test_texts = []
test_labels = []
original_scores = []

try:
    # 从url_list.txt读取数据
    with open('./url_list.txt', 'r', encoding='utf-8') as file:
        data = file.read()
    
    # 数据清洗，去除换行符
    data_cleaned = data.replace('\n', ' ')
    import ast
    data_list = ast.literal_eval(data_cleaned)
    
    # 处理所有样本，但最多取100个
    test_samples = data_list[:100] if len(data_list) > 100 else data_list
    for sample in test_samples:
        text = sample[0]
        original_score = sample[1]
        
        # 清洗文本
        text = re.sub(r'http\S+', '', text)
        text = ' '.join(text.split())
        
        test_texts.append(text)
        original_scores.append(original_score)
        # 将原始评分转换为情感类别
        test_labels.append(map_score_to_sentiment(original_score))
    
    print(f"成功加载{len(test_texts)}个测试样本")
except Exception as e:
    print(f"从文件加载数据时出错: {e}")
    # 使用一些预设的测试样本作为备用
    test_texts = ["我看过的最恶心的动漫，把我最喜欢的角色塑造成了纯纯的舔狗小丑，很难想象一个偶像企划会对角色有如此深的恶意，搞这种恶心玩意编剧全家没一个活人",  
                 "这光污染的画面，这廉价的特效，这残念的人设，这白开水的剧情……我也不知道我看这番是图个啥。"]
    original_scores = [1, 2]
    test_labels = [0, 0]  # 都映射为负面情感
    print("使用预设测试样本")

# 对测试数据进行分词处理
inputs = tokenizer(test_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
labels = torch.tensor(test_labels)

# 将输入数据移至相应设备
for key in inputs:
    inputs[key] = inputs[key].to(device)

# 将模型设置为评估模式
model.eval()

# 保存预测和实际标签
predictions_list = []
labels_list = labels.tolist()
predicted_texts = []

# 批处理预测以避免内存问题
batch_size = 8
num_samples = len(test_texts)

# 禁用梯度计算
with torch.no_grad():
    for i in range(0, num_samples, batch_size):
        # 获取当前批次数据
        batch_inputs = {k: v[i:i+batch_size] for k, v in inputs.items()}
        batch_labels = labels[i:i+batch_size]
        
        # 前向传播
        outputs = model(**batch_inputs)
        logits = outputs.logits
        
        # 获取预测结果
        batch_predictions = torch.argmax(logits, dim=1)
        predictions_list.extend(batch_predictions.cpu().tolist())
        
        # 保存文本和预测结果，用于详细分析
        for j, pred in enumerate(batch_predictions):
            idx = i + j
            if idx < len(test_texts):
                predicted_texts.append({
                    "text": test_texts[idx], 
                    "original_score": original_scores[idx],
                    "true_sentiment": labels_list[idx],
                    "predicted_sentiment": pred.item(),
                    "sentiment_match": labels_list[idx] == pred.item()
                })

# 计算指标
accuracy = accuracy_score(labels_list, predictions_list)
precision = precision_score(labels_list, predictions_list, average='macro', zero_division=0)
recall = recall_score(labels_list, predictions_list, average='macro', zero_division=0)
f1 = f1_score(labels_list, predictions_list, average='macro', zero_division=0)

# 确保输出结果
print("=" * 50)
print(f"测试样本数量: {len(test_texts)}")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1分数: {f1:.4f}")
print("=" * 50)

# 输出详细的分类报告
print("\n分类报告:")
print(classification_report(labels_list, predictions_list, 
                          target_names=['负面 (1-5分)', '正面 (6-10分)'],
                          zero_division=0))

# 分析错误预测
errors = [item for item in predicted_texts if not item["sentiment_match"]]
print(f"\n错误预测数量: {len(errors)}")

# 将预测结果保存到文件
with open("prediction_results.json", "w", encoding="utf-8") as f:
    json.dump(predicted_texts, f, ensure_ascii=False, indent=2)

print("预测结果已保存到 prediction_results.json")