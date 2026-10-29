"""LayerNorm rescue of the 2-4-1 sigmoid MLP; plot for 1.7 (BCE variant)."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10

sns.set_style("ticks")
sns.set_style("white")

paper_teal = "#4FB9AF"
paper_pink = "#FFACB7"
paper_indigo = "#08415C"
paper_coolgrey = "#C9C9D4"


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


X_all, y_all = make_moons(400, noise=0.25, seed=0)
X_tr = torch.tensor(X_all[:200], dtype=torch.float32)
y_tr = torch.tensor(y_all[:200], dtype=torch.float32)
X_te = torch.tensor(X_all[200:], dtype=torch.float32)
y_te = torch.tensor(y_all[200:], dtype=torch.float32)


class SigmoidMLP(nn.Module):
    """The 2-4-1 model from 1.6, with optional LayerNorm before the hidden
    sigmoid."""
    def __init__(self, scale=10.0, use_ln=False, seed=1):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W1 = nn.Parameter(torch.randn(2, 4, generator=g) * scale)
        self.b1 = nn.Parameter(torch.zeros(4))
        self.W2 = nn.Parameter(torch.randn(4, 1, generator=g) * scale)
        self.b2 = nn.Parameter(torch.zeros(1))
        self.use_ln = use_ln

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        if self.use_ln:
            mu = z1.mean(dim=1, keepdim=True)
            var = z1.var(dim=1, keepdim=True, unbiased=False)
            z1 = (z1 - mu) / torch.sqrt(var + 1e-5)
        h = torch.sigmoid(z1)
        z2 = h @ self.W2 + self.b2
        return torch.sigmoid(z2)


def bce(f, y):
    return F.binary_cross_entropy(f.squeeze(-1), y)


def train(use_ln, scale=10.0, epochs=5000, lr=0.01, seed=1):
    model = SigmoidMLP(scale=scale, use_ln=use_ln, seed=seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    tr, te = [], []
    for _ in range(epochs):
        opt.zero_grad()
        loss = bce(model(X_tr), y_tr)
        loss.backward()
        opt.step()
        tr.append(loss.item())
        with torch.no_grad():
            te.append(bce(model(X_te), y_te).item())
    return np.array(tr), np.array(te)


def train_with_grad_snapshot(use_ln, scale, epochs, lr, snap_epoch, seed=1):
    """Train, and at `snap_epoch` capture |grad W_1| and |grad W_2| per entry."""
    model = SigmoidMLP(scale=scale, use_ln=use_ln, seed=seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    tr, te = [], []
    snap_gW1 = snap_gW2 = None
    for ep in range(epochs):
        opt.zero_grad()
        loss = bce(model(X_tr), y_tr)
        loss.backward()
        if ep + 1 == snap_epoch:
            snap_gW1 = model.W1.grad.detach().flatten().abs().cpu().numpy()
            snap_gW2 = model.W2.grad.detach().flatten().abs().cpu().numpy()
        opt.step()
        tr.append(loss.item())
        with torch.no_grad():
            te.append(bce(model(X_te), y_te).item())
    return np.array(tr), np.array(te), snap_gW1, snap_gW2


SCALE = 2.0
EPOCHS = 5000
LR = 0.01

SNAP_EPOCH = 1000
trC,   teC,   gW1_no, gW2_no = train_with_grad_snapshot(
    False, SCALE, EPOCHS, LR, SNAP_EPOCH)
trCLN, teCLN, gW1_ln, gW2_ln = train_with_grad_snapshot(
    True,  SCALE, EPOCHS, LR, SNAP_EPOCH)

fig, axs = plt.subplots(1, 2, figsize=(9, 3))

plt.sca(axs[0])
epochs = np.arange(1, EPOCHS + 1)
plt.plot(epochs, trC,   color=paper_indigo, lw=1.5, label="no LN — train")
plt.plot(epochs, teC,   color=paper_indigo, lw=1.5, linestyle="--",
         label="no LN — test")
plt.plot(epochs, trCLN, color=paper_teal,   lw=1.5, label="LN — train")
plt.plot(epochs, teCLN, color=paper_teal,   lw=1.5, linestyle="--",
         label="LN — test")
plt.axvline(SNAP_EPOCH, color="lightgrey", lw=0.7, linestyle=":")
plt.xlabel("epoch")
plt.ylabel("loss (BCE)")
plt.title("training loss (large init)", fontsize=10)
plt.legend(frameon=False, fontsize=8, loc="upper right")
sns.despine()

ax = axs[1]
rng_j = np.random.RandomState(0)
jitter = 0.05
offset = 0.22
positions = {
    "no_W1": 0 - offset, "ln_W1": 0 + offset,
    "no_W2": 1 - offset, "ln_W2": 1 + offset,
}


def box(vals, pos, color):
    bp = ax.boxplot([vals], positions=[pos], widths=0.3,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=color, alpha=0.3,
                                  edgecolor=color, linewidth=1.0),
                    whiskerprops=dict(color=color, linewidth=1.0),
                    capprops=dict(color=color, linewidth=1.0),
                    medianprops=dict(color=color, linewidth=1.4))
    return bp


def pts(vals, pos, color):
    xs = pos + rng_j.uniform(-jitter, jitter, vals.size)
    ax.scatter(xs, vals, s=30, color=color, alpha=0.9,
               edgecolors="white", linewidths=0.4, zorder=3)


box(gW1_no, positions["no_W1"], paper_indigo)
box(gW2_no, positions["no_W2"], paper_indigo)
box(gW1_ln, positions["ln_W1"], paper_teal)
box(gW2_ln, positions["ln_W2"], paper_teal)

pts(gW1_no, positions["no_W1"], paper_indigo)
pts(gW2_no, positions["no_W2"], paper_indigo)
pts(gW1_ln, positions["ln_W1"], paper_teal)
pts(gW2_ln, positions["ln_W2"], paper_teal)

ax.set_xticks([0, 1])
ax.set_xticklabels([r"$W_1$", r"$W_2$"])
ax.set_xlim(-0.6, 1.6)
ax.set_yscale("log")
ax.set_ylabel(rf"$|\nabla_W L|$ per entry (epoch {SNAP_EPOCH})")
ax.set_title("gradient magnitudes by layer", fontsize=10)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=paper_indigo, alpha=0.4,
                         edgecolor=paper_indigo, label="no LN"),
                   Patch(facecolor=paper_teal, alpha=0.4,
                         edgecolor=paper_teal, label="LN")]
ax.legend(handles=legend_elements, frameon=False, fontsize=8,
          loc="upper left")
sns.despine()

plt.tight_layout()
plt.savefig("plot_ln_bce.pdf", dpi=300, bbox_inches="tight")

print(f"final loss: no-LN {trC[-1]:.4f}   LN {trCLN[-1]:.4f}")
print(f"final test loss: no-LN {teC[-1]:.4f}   LN {teCLN[-1]:.4f}")
print(f"|grad W_1| median  no-LN {np.median(gW1_no):.2e}   "
      f"LN {np.median(gW1_ln):.2e}")
print(f"|grad W_2| median  no-LN {np.median(gW2_no):.2e}   "
      f"LN {np.median(gW2_ln):.2e}")
