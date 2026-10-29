"""Plot df/dW vs W at x=1, b=0 for question 1.4."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10

sns.set_style("ticks")
sns.set_style("white")

paper_teal = "#4FB9AF"
paper_coolgrey = "#9596C6"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


WW = np.linspace(-8, 8, 400)
s = sigmoid(WW)
df_dW = s * (1 - s)

fig, ax = plt.subplots(figsize=(4, 3))
plt.plot(WW, df_dW, color=paper_teal, lw=1.8)
plt.xlabel("W")
plt.ylabel(r"$\partial f / \partial W$")
plt.ylim(-0.005, 0.28)
ax.set_yticks([])
sns.despine(left=False)
plt.tight_layout()
plt.savefig("plot_1_4.pdf", dpi=300, bbox_inches="tight")
