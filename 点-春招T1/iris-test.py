import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from random_forest2 import RandomForest
import sklearn

# 手动划分训练集和测试集
def train_test_split_manual(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test

# 计算准确率
def calculate_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# 计算精确率
def calculate_precision(y_true, y_pred, num_classes):
    precision_scores = []
    for c in range(num_classes):
        true_positives = np.sum((y_true == c) & (y_pred == c))
        predicted_positives = np.sum(y_pred == c)
        if predicted_positives == 0:
            precision = 0
        else:
            precision = true_positives / predicted_positives
        precision_scores.append(precision)
    return np.mean(precision_scores)

# 计算召回率
def calculate_recall(y_true, y_pred, num_classes):
    recall_scores = []
    for c in range(num_classes):
        true_positives = np.sum((y_true == c) & (y_pred == c))
        actual_positives = np.sum(y_true == c)
        if actual_positives == 0:
            recall = 0
        else:
            recall = true_positives / actual_positives
        recall_scores.append(recall)
    return np.mean(recall_scores)

# 计算 F1 分数
def calculate_f1_score(y_true, y_pred, num_classes):
    precision = calculate_precision(y_true, y_pred, num_classes)
    recall = calculate_recall(y_true, y_pred, num_classes)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# 计算混淆矩阵
def calculate_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i]][y_pred[i]] += 1
    return cm

# 加载鸢尾花数据集
iris = fetch_ucirepo(id=53)
X = iris.data.features.values
y = iris.data.targets.values.ravel()

# 不用sklearn-label，手动解码
y[y == 'Iris-setosa'] = 0
y[y == 'Iris-versicolor'] = 1
y[y == 'Iris-virginica'] = 2
y = y.astype(int)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_state=42)

# 检查类别分布
print("Training set class distribution:", np.bincount(y_train))
print("Test set class distribution:", np.bincount(y_test))

# 初始化并训练随机森林模型
rf = RandomForest(n_trees=10)  # 假设构造函数需要指定树的数量
rf.fit(X_train, y_train)

# 进行预测
y_pred = rf.predict(X_test)

# 检查预测结果
print("y_test:", y_test)
print("y_pred:", y_pred)

# 计算评估指标
num_classes = len(np.unique(y))
accuracy = calculate_accuracy(y_test, y_pred)
precision = calculate_precision(y_test, y_pred, num_classes)
recall = calculate_recall(y_test, y_pred, num_classes)
f1 = calculate_f1_score(y_test, y_pred, num_classes)

# 输出结果分析
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# 可视化混淆矩阵
cm = calculate_confusion_matrix(y_test, y_pred, num_classes)
print("Confusion Matrix:\n", cm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
            yticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 特征重要性分析
feature_importance = rf.feature_importance()
feature_names = iris.data.features.columns
indices = np.argsort(feature_importance)
plt.figure()
plt.title("Feature Importance")
plt.barh(range(len(indices)), feature_importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()
"""
except AttributeError:
    print("The RandomForest class does not have a feature_importance method.")
"""