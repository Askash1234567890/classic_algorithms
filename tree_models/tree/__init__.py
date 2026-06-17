import numpy as np
from collections import Counter

class Node():
    def __init__(
            self,
            feature,
            threshold,
            left,
            right,
            value = None
        ):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value

class DecisionTreeClassifier():
    def __init__(self, max_depth=4):
        self.root      = None
        self.max_depth = max_depth

    def _best_split(self, x: np.array, y: np.array):
        n, n_features = x.shape
        ones = int(y.sum())
        best_gini = float("inf")
        best_feature, best_threshold = None, None

        for feature in range(n_features):
            order = np.argsort(x[:, feature])
            x_sorted = x[order]
            y_sorted = y[order]
            left_label, right_label = [0, 0], [n - ones, ones]

            for i in range(n - 1):
                label = y_sorted[i][0]
                left_label[label] += 1
                right_label[label] -= 1

                gini_left  = 1 - (left_label[0]  / (i + 1))     ** 2 - (left_label[1]  / (i + 1))     ** 2
                gini_right = 1 - (right_label[0] / (n - i - 1)) ** 2 - (right_label[1] / (n - i - 1)) ** 2
                weighted_gini = (gini_left * (i + 1) + gini_right * (n - i - 1)) / n

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = (x_sorted[i, feature] + x_sorted[i + 1, feature]) / 2

        return best_feature, best_threshold


    def _build_tree(self, x: np.array, y: np.array, depth: int):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            maj = Counter(y.flatten()).most_common(1)[0][0]
            return Node(feature=None, threshold=None, left=None, right=None, value=maj)
        
        best_feature, best_threshold = self._best_split(x=x, y=y)
        if best_feature is None:
            maj = Counter(y.flatten()).most_common(1)[0][0]
            return Node(feature=None, threshold=None, left=None, right=None, value=maj)

        mask = x[:, best_feature] <= best_threshold

        left  = self._build_tree(x=x[mask], y=y[mask], depth=depth + 1)
        right = self._build_tree(x=x[~mask], y=y[~mask], depth=depth + 1)

        return Node(
            feature   = best_feature,
            threshold = best_threshold,
            left      = left,
            right     = right
        )

    def _gini(self, y: np.array):
        cnt = Counter(y.flatten())
        res = 1
        for _, v in cnt.items():
            res -= (v / y.shape[0]) ** 2
        return res

    def fit(self, x: np.array, y: np.array):
        self.root = self._build_tree(x=x, y=y, depth=0)

    def _predict_one(self, node: Node, x: np.array):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_one(node=node.left, x=x)
        else:
            return self._predict_one(node=node.right, x=x)
        
    def predict(self, x: np.array):
        return np.array([self._predict_one(x=sample, node=self.root) for sample in x])
    

class DecisionTreeRegressor():
    def __init__(self, max_depth=4, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=1e-4):
        self.max_depth            = max_depth
        self.min_samples_split    = min_samples_split
        self.min_samples_leaf     = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.root                 = None
        self.root_var             = None
    
    def _best_split(self, x: np.array, y: np.array):
        best_var = float("inf")
        best_feature, best_threshold = None, None
        n, n_features = x.shape
        for feature in range(n_features):
            order = np.argsort(x[:, feature])
            x_ordered = x[order]
            y_ordered = y[order]

            sum_left, sum_sq_left = 0, 0
            sum_right, sum_sq_right = y.sum(), (y * y).sum()

            mean_left = 0.0
            mean_right = sum_right / n

            left_var, right_var = 0, sum_sq_right / n - mean_right ** 2

            for row in range(n - 1):
                sum_left += y_ordered[row][0]
                sum_sq_left += y_ordered[row][0] ** 2
                mean_left = sum_left / (row + 1)

                sum_right -= y_ordered[row][0]
                sum_sq_right -= y_ordered[row][0] ** 2
                mean_right = sum_right / (n - row - 1)

                n_left, n_right = row + 1, n - row - 1
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                left_var, right_var = sum_sq_left / n_left - mean_left ** 2, sum_sq_right / n_right - mean_right ** 2
                var = (left_var * n_left + right_var * n_right) / n

                if var < best_var:
                    best_var = var
                    best_feature = feature
                    best_threshold = (x_ordered[row, feature] + x_ordered[row + 1, feature]) / 2

        return best_feature, best_threshold
    
    def _build_tree(self, x: np.array, y: np.array, depth: int):
        n = y.shape[0]
        impurity_decrease = float(np.var(y)) / self.root_var
        if depth >= self.max_depth or n < self.min_samples_split or impurity_decrease <= self.min_impurity_decrease:
            return Node(threshold=None, feature=None, value=float(np.mean(y)), left=None, right=None)

        feature, threshold = self._best_split(x=x, y=y)
        mask = x[:, feature] <= threshold

        left  = self._build_tree(x=x[mask], y=y[mask], depth=depth + 1)
        right = self._build_tree(x=x[~mask], y=y[~mask], depth=depth + 1)

        return Node(
            feature=feature,
            threshold=threshold,
            value=None,
            left=left,
            right=right
        )

    def fit(self, x: np.array, y: np.array):
        self.root_var = float(np.var(y))
        self.root = self._build_tree(x=x, y=y, depth=0)

    def _predict_one(self, x: np.array, node: Node):
        if node.value is not None:
            return node.value
        
        feature, threshold = node.feature, node.threshold
        if x[feature] <= threshold:
            return self._predict_one(node=node.left, x=x)
        else:
            return self._predict_one(node=node.right, x=x)

    def predict(self, x: np.array):
        return np.array([self._predict_one(x=sample, node=self.root) for sample in x])