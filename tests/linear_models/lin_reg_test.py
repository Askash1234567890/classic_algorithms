import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from linear_models import LinearRegression

np.random.seed(42)
n, d = 200, 3
TRUE_W = np.array([2.0, -1.5, 0.8])
TRUE_B = 3.0

x = np.random.randn(n, d)
y = (x @ TRUE_W + TRUE_B + np.random.randn(n) * 0.3).reshape(-1, 1)

model = LinearRegression(L1=0.0, L2=0.0, epochs=300, lr=0.1)
model.fit(x, y)

pred = model.predict(x)
print(f"R² = {r2_score(y, pred):.4f}")
print(f"True weights:    {TRUE_W}, bias: {TRUE_B}")
print(f"Learned weights: {model.weights[:-1].ravel()}, bias: {model.weights[-1, 0]:.4f}")

history = np.array(model.weight_history).squeeze()
epochs = np.arange(len(history))
labels = [f"w{i} (true={TRUE_W[i]})" for i in range(d)] + [f"bias (true={TRUE_B})"]
true_vals = list(TRUE_W) + [TRUE_B]

fig, axes = plt.subplots(d + 1, 1, figsize=(9, 2.5 * (d + 1)), sharex=True)
for i, (ax, label, tv) in enumerate(zip(axes, labels, true_vals)):
    ax.plot(epochs, history[:, i], label="learned")
    ax.axhline(tv, color="red", linestyle="--", label=f"true = {tv}")
    ax.set_ylabel(label, fontsize=9)
    ax.legend(fontsize=8)

axes[-1].set_xlabel("epoch")
fig.suptitle("Coefficient evolution during gradient descent", fontsize=12)
plt.tight_layout()
plt.show()
plt.close()
