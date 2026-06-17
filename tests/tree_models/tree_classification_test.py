import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as SklearnTree

from tree_models.tree import DecisionTreeClassifier

np.random.seed(42)
x, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0)
y = y.reshape(-1, 1)

model = DecisionTreeClassifier(max_depth=4)
model.fit(x, y)
pred = model.predict(x)

sklearn_model = SklearnTree(max_depth=4)
sklearn_model.fit(x, y)
sklearn_pred = sklearn_model.predict(x)

print(f"My tree accuracy:      {accuracy_score(y, pred):.4f}")
print(f"Sklearn tree accuracy: {accuracy_score(y, sklearn_pred):.4f}")

h = 0.05
x0_min, x0_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
x1_min, x1_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h))
grid = np.c_[xx0.ravel(), xx1.ravel()]

Z_my      = model.predict(grid).reshape(xx0.shape)
Z_sklearn = sklearn_model.predict(grid).reshape(xx0.shape)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, Z, title in zip(axes, [Z_my, Z_sklearn], ["My DecisionTree", "Sklearn DecisionTree"]):
    ax.contourf(xx0, xx1, Z, alpha=0.3, cmap="RdBu")
    ax.scatter(x[:, 0], x[:, 1], c=y.ravel(), cmap="RdBu", edgecolors="k", s=20)
    ax.set_title(title)
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")

plt.tight_layout()
plt.show()
plt.close()
