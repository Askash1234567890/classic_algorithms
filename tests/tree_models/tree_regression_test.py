import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor as SklearnRegressor

from tree_models.tree import DecisionTreeRegressor

np.random.seed(42)
x, y = make_regression(n_samples=300, n_features=1, noise=15)
y = y.reshape(-1, 1)

model = DecisionTreeRegressor(max_depth=4)
model.fit(x, y)
pred = model.predict(x)

sklearn_model = SklearnRegressor(max_depth=4)
sklearn_model.fit(x, y)
sklearn_pred = sklearn_model.predict(x)

print(f"My tree R²:      {r2_score(y, pred):.4f}")
print(f"Sklearn tree R²: {r2_score(y, sklearn_pred):.4f}")

# --- visualize predictions vs ground truth ---
order = np.argsort(x[:, 0])
x_sorted = x[order, 0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, p, title in zip(axes, [pred[order], sklearn_pred[order]], ["My DecisionTreeRegressor", "Sklearn DecisionTreeRegressor"]):
    ax.scatter(x_sorted, y[order], s=15, alpha=0.5, label="data")
    ax.step(x_sorted, p, color="red", linewidth=2, label="prediction")
    ax.set_title(f"{title}\nR² = {r2_score(y, p if p is pred[order] else sklearn_pred):.4f}")
    ax.set_xlabel("feature 0")
    ax.set_ylabel("target")
    ax.legend()

plt.tight_layout()
plt.show()
plt.close()
