"""Train a 2-4-1 sigmoid MLP on two-moons; plots for 1.6."""
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


def init_params(d, h, scale, seed=1):
    rng = np.random.RandomState(seed)
    if scale == 'xavier':
        W1 = rng.randn(d, h)
        W2 = rng.randn(h, 1)
    elif scale == 'huge':
        W1 = rng.randn(d, h) * 10.0
        W2 = rng.randn(h, 1) * 10.0
    elif scale == 'zero':
        W1 = np.zeros((d, h))
        W2 = np.zeros((h, 1))
    elif scale == 'ones':
        W1 = np.ones((d, h))
        W2 = np.ones((h, 1))
    return {'W1': W1, 'b1': np.zeros(h), 'W2': W2, 'b2': np.zeros(1)}


def forward(X, p):
    z1 = X @ p['W1'] + p['b1']; h = sigmoid(z1)
    z2 = h @ p['W2'] + p['b2']; f = sigmoid(z2)
    return f, (z1, h, z2, f)


def loss(f, y):
    return np.mean((f.ravel() - y) ** 2)


def backward(X, y, p, cache):
    z1, h, z2, f = cache
    n = X.shape[0]
    y = y.reshape(-1, 1)
    dLdf = 2 * (f - y) / n
    dLdz2 = dLdf * f * (1 - f)
    dW2 = h.T @ dLdz2; db2 = dLdz2.sum(0)
    dLdh = dLdz2 @ p['W2'].T
    dLdz1 = dLdh * h * (1 - h)
    dW1 = X.T @ dLdz1; db1 = dLdz1.sum(0)
    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}


def train(X_tr, y_tr, X_te, y_te, scale, lr=1.0, epochs=800, hidden=4,
          init_seed=1):
    p = init_params(X_tr.shape[1], hidden, scale, init_seed)
    tr, te = [], []
    for _ in range(epochs):
        f, cache = forward(X_tr, p)
        g = backward(X_tr, y_tr, p, cache)
        for k in p:
            p[k] = p[k] - lr * g[k]
        tr.append(loss(f, y_tr))
        f_te, _ = forward(X_te, p)
        te.append(loss(f_te, y_te))
    return np.array(tr), np.array(te)


X_all, y_all = make_moons(400, noise=0.25, seed=0)
X_tr, y_tr = X_all[:200], y_all[:200]
X_te, y_te = X_all[200:], y_all[200:]

EPOCHS = 5000
LR = 0.01
trA, teA = train(X_tr, y_tr, X_te, y_te, 'zero',   lr=LR, epochs=EPOCHS)
trB, teB = train(X_tr, y_tr, X_te, y_te, 'xavier', lr=LR, epochs=EPOCHS)
trC, teC = train(X_tr, y_tr, X_te, y_te, 'huge',   lr=LR, epochs=EPOCHS)
trD, teD = train(X_tr, y_tr, X_te, y_te, 'ones',   lr=LR, epochs=EPOCHS)

# ----- moons scatter -----
fig, ax = plt.subplots(figsize=(3.2, 3.0))
m0 = y_all == 0
m1 = y_all == 1
plt.scatter(X_all[m0, 0], X_all[m0, 1], s=10, color=paper_teal, alpha=0.75,
            label="class 0")
plt.scatter(X_all[m1, 0], X_all[m1, 1], s=10, color=paper_pink, alpha=0.75,
            label="class 1")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.legend(frameon=False, fontsize=8, loc="upper right")
sns.despine()
plt.tight_layout()
plt.savefig("plot_moons.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ----- four training curves -----
panels = [
    ("A — zero", trA, teA),
    ("B — Gaussian", trB, teB),
    ("C — large magnitude", trC, teC),
    (r"D — symmetric ($W_{ij}=1$)", trD, teD),
]

fig, axs = plt.subplots(1, 4, figsize=(13, 3), sharex=True, sharey=True)
axs = axs.flatten()
for i, (label, tr, te) in enumerate(panels):
    plt.sca(axs[i])
    epochs = np.arange(1, len(tr) + 1)
    plt.plot(epochs, tr, color=paper_teal, lw=1.5, label="train")
    plt.plot(epochs, te, color=paper_pink, lw=1.5, linestyle="--", label="test")
    plt.title(label, fontsize=10)
    plt.xlabel("epoch")
    if i == 0:
        plt.ylabel("loss")
    if i == 3:
        plt.legend(frameon=False, fontsize=9, loc="upper right")
    sns.despine()

plt.tight_layout()
plt.savefig("plot_1_5.pdf", dpi=300, bbox_inches="tight")
