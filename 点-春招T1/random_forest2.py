import numpy as np
from collections import Counter
"""

#此为AI改后的代码，但有问题。
#底下是进行修改部分，改后在下面的能跑通
#feature_importance(self):
# print(self.trees[0].feature_importances.shape[0])
#数据类型错误了，树如果节点存的不是向量无法显示，转链表
#        tree=self.trees[0]
#        importance=np.zeros(tree.tree[0])


class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  # 特征子集的大小
        self.tree = None

    def fit(self, X, y, depth=0):
        # print(f"Depth {depth}: len(y) = {len(y)}")
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # 递归终止条件
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1:
            # print(f"Stopping at depth {depth}")
            return
        # 随机选择特征子集
        if self.n_features is None:
            self.n_features = n_features
        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)
        # 选择最佳特征和阈值进行分裂
        best_feature, best_threshold = self._best_split(X, y, feature_idxs)

        # 如果没有找到有效的分裂，停止递归
        if best_feature is None:
            # print(f"No valid split at depth {depth}")
            return

        # 分裂数据
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        # print(f"Depth {depth}: Split on feature {best_feature} with threshold {best_threshold}")
        # print(f"Left node samples: {len(y_left)}, Right node samples: {len(y_right)}")

        # 递归训练左右子树
        self.fit(X_left, y_left, depth + 1)
        self.fit(X_right, y_right, depth + 1)

    def _best_split(self, X, y, feature_idxs):
        best_feature_idx, best_threshold, best_gain = None, None, -1

        for feature_idx in feature_idxs:
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:

                gain = self._information_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def _information_gain(self, X, y, feature_idx, threshold):
        left_indices = X[:, feature_idx] <= threshold
        right_indices = X[:, feature_idx] > threshold

        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(y[left_indices]), len(y[right_indices])
        e_left, e_right = self._entropy(y[left_indices]), self._entropy(y[right_indices])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        parent_entropy = self._entropy(y)

        return parent_entropy - child_entropy

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):

        counter = Counter(y)
        # print("y:",y)
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
"""
import numpy as np
from collections import Counter
class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  # 特征子集的大小
        self.tree = None

    def fit(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # 递归终止条件
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1:
            leaf_value = self._most_common_label(y)
            return leaf_value

        # 随机选择特征子集
        if self.n_features is None:
            self.n_features = n_features
        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)

        # 选择最佳特征和阈值进行分裂
        best_feature, best_threshold = self._best_split(X, y, feature_idxs)

        # 如果没有找到有效的分裂，返回叶子节点
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return leaf_value

        # 分裂数据
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        # 递归训练左右子树
        left_subtree = self.fit(X_left, y_left, depth + 1)
        right_subtree = self.fit(X_right, y_right, depth + 1)

        # 构建树结构
        self.tree = (best_feature, best_threshold, left_subtree, right_subtree)
        return self.tree

    def _best_split(self, X, y, feature_idxs):
        best_feature_idx, best_threshold, best_gain = None, None, -1

        for feature_idx in feature_idxs:
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def _information_gain(self, X, y, feature_idx, threshold):
        left_indices = X[:, feature_idx] <= threshold
        right_indices = X[:, feature_idx] > threshold

        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(y[left_indices]), len(y[right_indices])
        e_left, e_right = self._entropy(y[left_indices]), self._entropy(y[right_indices])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        parent_entropy = self._entropy(y)

        return parent_entropy - child_entropy

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

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
            return node  # 叶子节点直接返回类别标签

class RandomForest:
    def __init__(self, n_trees=100, max_depth=100, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  # 每棵树使用的特征子集大小
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            # Bootstrap采样
            X_sample, y_sample = self._bootstrap_sample(X, y)

            # 训练单棵树
            tree.tree = tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)  # 有放回采样
        return X[idxs], y[idxs]

    def predict(self, X):
        # 每棵树的预测结果
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # 多数投票
        tree_preds = np.swapaxes(tree_preds, 0, 1)  # 转置，方便按样本统计
        y_pred = [self._most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

    def _most_common_label(self, y):
        # print("y:",y)
        counter = Counter(y)
        return counter.most_common(1)[0][0]


    def feature_importance(self):
       # print(self.trees[0].feature_importances.shape[0])
        if not self.trees:
            return None
        tree=self.trees[0]
        importance=np.zeros(tree.tree[0])

        for tree in self.trees:
            importance += tree.tree[0]

        # 归一化特征重要性
        total_importance = importance.sum()
        if total_importance > 0:
            importance /= total_importance
        return importance