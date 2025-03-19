import graphviz
import numpy as np
from collections import Counter
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
# 计算熵
def entropy(x):
    hist = np.bincount(x)
    ps = hist / len(x)
    return -np.sum([p * np.log2(p) for p in ps if (p > 0)])

# 计算信息增益
def information_gain(X, y, feature_idx, threshold):
    left_indices = X[:, feature_idx] <= threshold
    right_indices = X[:, feature_idx] > threshold
    n = len(y)
    n_left, n_right = len(y[left_indices]), len(y[right_indices])
    e_left, e_right = entropy(y[left_indices]), entropy(y[right_indices])
    sub = (n_left / n) * e_left + (n_right / n) * e_right
    pre = entropy(y)
    return pre - sub

# 寻找最佳分割点
def best_split(X, y):
    best_feature_idx, best_threshold, best_gain = None, None, -1
    n_features = X.shape[1]
    for feature_idx in range(n_features):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            gain = information_gain(X, y, feature_idx, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature_idx = feature_idx
                best_threshold = threshold
    return best_feature_idx, best_threshold, best_gain

# 决策树类
class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        # 停止条件
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return leaf_value
        # 找到最佳分割
        feature_idx, threshold, _ = best_split(X, y)
        if feature_idx is None:
            return self._most_common_label(y)
        # 递归构建子树
        left_indices = X[:, feature_idx] <= threshold
        right_indices = X[:, feature_idx] > threshold
        left_subtree = self.fit(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self.fit(X[right_indices], y[right_indices], depth + 1)
        if depth == 0:
            self.tree = (feature_idx, threshold, left_subtree, right_subtree)
        return (feature_idx, threshold, left_subtree, right_subtree)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if isinstance(node, tuple):
            feature_idx, threshold, left_subtree, right_subtree = node
            if x[feature_idx] <= threshold:
                return self._traverse_tree(x, left_subtree)
            else:
                return self._traverse_tree(x, right_subtree)
        else:
            return node

    def visualize_tree(self, feature_names=None, class_names=None):
        dot = graphviz.Digraph()
        def add_nodes_edges(node, parent_id=None, is_left=None):
            if isinstance(node, tuple):
                feature_idx, threshold, left_subtree, right_subtree = node
                node_id = str(id(node))
                if feature_names:
                    label = f"Feature: {feature_names[feature_idx]}\nThreshold: {threshold}"
                else:
                    label = f"Feature {feature_idx}\nThreshold: {threshold}"
                dot.node(node_id, label)
                if parent_id:
                    if is_left:
                        dot.edge(parent_id, node_id, label="<= Threshold")
                    else:
                        dot.edge(parent_id, node_id, label="> Threshold")
                add_nodes_edges(left_subtree, node_id, True)
                add_nodes_edges(right_subtree, node_id, False)
            else:
                node_id = str(id(node))
                if class_names:
                    label = f"Class: {class_names[node]}"
                else:
                    label = f"Class: {node}"
                dot.node(node_id, label)
                if parent_id:
                    if is_left:
                        dot.edge(parent_id, node_id, label="<= Threshold")
                    else:
                        dot.edge(parent_id, node_id, label="> Threshold")
        add_nodes_edges(self.tree)
        return dot

# 从 UCI 仓库获取鸢尾花数据集
iris = fetch_ucirepo(id=53)
X = iris.data.features.values
y = iris.data.targets.values.ravel()

# 对目标变量进行编码，转换为整数类型
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

feature_names = iris.data.features.columns.tolist()
class_names = label_encoder.classes_.tolist()

# 训练决策树
tree = DecisionTree(max_depth=3)
tree.fit(X, y)
y_pred = tree.predict(X)
accuracy = np.sum(y_pred == y) / len(y)
print(f"Accuracy: {accuracy}")

# 可视化决策树
dot = tree.visualize_tree(feature_names=feature_names, class_names=class_names)
dot.render('decision_tree', format='png', cleanup=True, view=True)