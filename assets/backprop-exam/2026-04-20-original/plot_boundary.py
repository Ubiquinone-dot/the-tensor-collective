"""Decision boundary: Gaussian-init MLP (1.6B) vs KNN; plot for 1.8."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10

sns.set_style("ticks")
sns.set_style("white")

paper_teal = "#4FB9AF"
paper_pink = "#FFACB7"


def make_moons(n, noise=0.2, seed=0):
    rng = np.random.RandomState(seed)
    n1 = n // 2; n2 = n - n1
    t1 = rng.uniform(0, np.pi, n1)
    X1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    t2 = rng.uniform(0, np.pi, n2)
    X2 = np.stack([1 - np.cos(t2), -np.sin(t2) + 0.5], axis=1)
    X = np.concatenate([X1, X2], axis=0) + rng.randn(n, 2) * noise
    y = np.concatenate([np.zeros(n1), np.ones(n2)])
    idx = rng.permutation(n)
    return X[idx], y[idx]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def train_mlp(X_tr, y_tr, lr=1.0, epochs=30000, hidden=4, seed=1):
    """Gaussian-init 2-4-1 with MSE + sigmoid; trained to convergence so the
    decision boundary for 1.8 is well-fit. Same architecture/init family as
    1.6(B), but with lr/epochs cranked — MSE + sigmoid output has a vanishing
    sigma' factor at the output, so 5000 × lr=0.01 under-trains visibly."""
    rng = np.random.RandomState(seed)
    d = X_tr.shape[1]
    p = {
        'W1': rng.randn(d, hidden),
        'b1': np.zeros(hidden),
        'W2': rng.randn(hidden, 1),
        'b2': np.zeros(1),
    }
    n = X_tr.shape[0]
    y = y_tr.reshape(-1, 1)
    for _ in range(epochs):
        z1 = X_tr @ p['W1'] + p['b1']
        h = sigmoid(z1)
        z2 = h @ p['W2'] + p['b2']
        f = sigmoid(z2)
        dLdz2 = 2 * (f - y) / n * f * (1 - f)
        dW2 = h.T @ dLdz2; db2 = dLdz2.sum(0)
        dLdh = dLdz2 @ p['W2'].T
        dLdz1 = dLdh * h * (1 - h)
        dW1 = X_tr.T @ dLdz1; db1 = dLdz1.sum(0)
        p['W1'] -= lr * dW1; p['b1'] -= lr * db1
        p['W2'] -= lr * dW2; p['b2'] -= lr * db2
    return p


def mlp_predict(X, p):
    z1 = X @ p['W1'] + p['b1']
    h = sigmoid(z1)
    z2 = h @ p['W2'] + p['b2']
    return sigmoid(z2).ravel()


def knn_predict(X_query, X_train, y_train, k=5):
    preds = np.zeros(X_query.shape[0])
    for i, q in enumerate(X_query):
        d = np.linalg.norm(X_train - q, axis=1)
        idx = np.argsort(d)[:k]
        preds[i] = y_train[idx].mean()
    return preds


X_all, y_all = make_moons(400, noise=0.25, seed=0)
X_tr, y_tr = X_all[:200], y_all[:200]

p = train_mlp(X_tr, y_tr, lr=1.0, epochs=30000, hidden=4, seed=1)

# sanity print
f_tr = mlp_predict(X_tr, p)
mlp_acc = np.mean((f_tr > 0.5) == y_tr.astype(bool))
print(f"MLP train accuracy: {mlp_acc:.3f}   train MSE: {np.mean((f_tr - y_tr)**2):.4f}")

pad = 0.4
xx, yy = np.meshgrid(
    np.linspace(X_all[:, 0].min() - pad, X_all[:, 0].max() + pad, 300),
    np.linspace(X_all[:, 1].min() - pad, X_all[:, 1].max() + pad, 300))
grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

mlp_prob    = mlp_predict(grid, p).reshape(xx.shape)
knn1_prob   = knn_predict(grid, X_tr, y_tr, k=1  ).reshape(xx.shape)
knn5_prob   = knn_predict(grid, X_tr, y_tr, k=5  ).reshape(xx.shape)
knn200_prob = knn_predict(grid, X_tr, y_tr, k=200).reshape(xx.shape)

print(f"class balance in train set: class 0 = {int((y_tr == 0).sum())}, "
      f"class 1 = {int((y_tr == 1).sum())}")
print(f"knn K=200 mean prob over grid: {knn200_prob.mean():.3f}")

# Shuffled labels — students must reason from boundary shape, not position.
panels = [
    ("Model A", knn5_prob),
    ("Model B", mlp_prob),
    ("Model C", knn1_prob),
    ("Model D", knn200_prob),
]

fig, axs = plt.subplots(1, 4, figsize=(12, 3.0), sharex=True, sharey=True)
axs = axs.flatten()
for i, (label, prob) in enumerate(panels):
    plt.sca(axs[i])
    plt.contourf(xx, yy, prob, levels=[-0.01, 0.5, 1.01],
                 colors=[paper_teal, paper_pink], alpha=0.2)
    plt.contour(xx, yy, prob, levels=[0.5],
                colors="black", linewidths=1.0)
    m0 = y_tr == 0
    m1 = y_tr == 1
    plt.scatter(X_tr[m0, 0], X_tr[m0, 1], s=10, color=paper_teal,
                edgecolors="white", linewidths=0.3)
    plt.scatter(X_tr[m1, 0], X_tr[m1, 1], s=10, color=paper_pink,
                edgecolors="white", linewidths=0.3)
    plt.title(label, fontsize=10)
    plt.xlabel(r"$x_1$")
    if i == 0:
        plt.ylabel(r"$x_2$")
    sns.despine()

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_handles = [
    Patch(facecolor=paper_teal, alpha=0.3, label="predicted class 0"),
    Patch(facecolor=paper_pink, alpha=0.3, label="predicted class 1"),
    Line2D([], [], marker="o", linestyle="none",
           markerfacecolor=paper_teal, markeredgecolor="white",
           markeredgewidth=0.4, markersize=7, label="training class 0"),
    Line2D([], [], marker="o", linestyle="none",
           markerfacecolor=paper_pink, markeredgecolor="white",
           markeredgewidth=0.4, markersize=7, label="training class 1"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.savefig("plot_boundary.pdf", dpi=300, bbox_inches="tight")
