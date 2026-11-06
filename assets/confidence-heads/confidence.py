"""Confidence panels: heads trained only on structure prediction and self-error
estimation, or on masked-token prediction, discriminate experimental outcomes they
were never trained on.

Panel a  pLDDT vs curated disorder: distribution split by the experimental label.
Panel b  one ROC AUC per task, for the AlphaFold confidence head and for ESM
         sequence likelihood. Open points are individual assays, filled points the
         median across assays.

Sources, in the order they appear in panel b (see EVIDENCE.md for detail):
  disorder            recomputed here from BioComputingUP/af2_af3_idr_analysis
                      (CAID3 Disorder-PDB, DisProt labels; 319 proteins, 99,239
                      assessed residues) -> AUC 0.933
  PPI vs decoy        PPIscreenML (eLife reviewed preprint 98179), AF2 default
                      confidence on a DOCKGROUND benchmark -> AUC 0.84
  binder binds        recomputed here from adaptyvbio/egfr_competition_2 open
                      results (BLI, 378 designs, 53 binders) -> ipTM AUC 0.636
  pathogenicity       Brandes et al. 2023 Nat Genet, ESM1b on ClinVar -> AUC 0.905
  DMS categories      ProteinGym v1.3 zero-shot substitution AUC, ESM2 650M, per
                      assay, grouped by coarse selection type

Run: python3 confidence.py  ->  out/fig-confidence.svg (+ light/dark check PNGs)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
for f in fm.findSystemFonts():
    if "Inter" in f:
        fm.fontManager.addfont(f)

INK = "#8a8378"
POS = "#d9673f"   # AlphaFold confidence head
NEG = "#1c9a85"   # ESM likelihood
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 11, "text.color": INK,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "mathtext.fontset": "cm", "svg.fonttype": "path", "figure.facecolor": "none",
    "axes.facecolor": "none", "savefig.transparent": True, "svg.hashsalt": "confidence"})


def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(1.0)


def tag(ax, s, x=0, y=1.02):
    return ax.text(x, y, s, transform=ax.transAxes, fontsize=15, weight="bold",
                   va="bottom", color=INK)


res = pd.read_csv(os.path.join(DATA, "AF2_res.csv.gz"))
auc_disorder = roc_auc_score(res.dis, -res.plddt)
pg = pd.read_csv(os.path.join(DATA, "proteingym_auc.csv"))
PGCOL = "ESM2 (650M)"

# literature values that are not recomputed here
AUC_PPI = 0.84          # PPIscreenML, AF2 default confidence, DOCKGROUND
AUC_BINDER = 0.636      # recomputed, Adaptyv EGFR round 2 ipTM
AUC_CLINVAR = 0.905     # Brandes et al. 2023, ESM1b on ClinVar

# task, colour, single AUC, per-assay points
rows = [
    ("disorder",          POS, auc_disorder, None),
    ("interaction",       POS, AUC_PPI, None),
    ("binder binds",      POS, AUC_BINDER, None),
    ("pathogenicity",     NEG, AUC_CLINVAR, None),
    ("stability",         NEG, None, "Stability"),
    ("activity",          NEG, None, "Activity"),
    ("fitness",           NEG, None, "OrganismalFitness"),
    ("expression",        NEG, None, "Expression"),
    ("binding affinity",  NEG, None, "Binding"),
]

fig = plt.figure(figsize=(10.4, 3.6))
G = fig.add_gridspec(1, 2, left=.065, right=.99, bottom=.16, top=.88,
                     wspace=.42, width_ratios=[1, 1.35])

# ---- a. the two experimental classes seen through the confidence head
ax = fig.add_subplot(G[0]); ax.set_gid("panel-plddt-split"); clean(ax); tag(ax, "a", 0, .985)
bins = np.arange(0, 101, 2.5)
for lab, c, name in ((0, NEG, "ordered"), (1, POS, "disordered")):
    v = res.plddt[res.dis == lab]
    h, _ = np.histogram(v, bins=bins, density=True)
    ax.fill_between(bins[:-1], 0, h, step="post", color=c, alpha=.30, lw=0)
    ax.step(bins[:-1], h, where="post", color=c, lw=1.4, label=name)
ax.set_xlim(0, 100); ax.set_ylim(0, None)
ax.set_xlabel("pLDDT"); ax.set_ylabel("density")
ax.set_yticks([])
leg = ax.legend(frameon=False, loc="upper left", handlelength=1.1, fontsize=10.5)
for t, c in zip(leg.get_texts(), (NEG, POS)):
    t.set_color(c)
ax.text(.52, .88, f"AUC = {auc_disorder:.2f}", transform=ax.transAxes, ha="left",
        fontsize=11, color=INK)

# ---- b. one AUC per task
ax = fig.add_subplot(G[1]); ax.set_gid("panel-task-auc"); clean(ax); tag(ax, "b", 0, .985)
ax.spines["left"].set_visible(False)
ys = np.arange(len(rows))[::-1]
rng = np.random.default_rng(3)
for y, (name, c, val, cat) in zip(ys, rows):
    ax.plot([.5, 1.0], [y, y], color=INK, lw=.5, alpha=.22, zorder=0)
    if cat is not None:
        v = pg.loc[pg.coarse_selection_type == cat, PGCOL].dropna().values
        ax.scatter(v, y + rng.uniform(-.17, .17, v.size), s=13, c=c, alpha=.30,
                   lw=0, zorder=2)
        val = float(np.median(v))
    ax.scatter([val], [y], s=62, c=c, zorder=3, lw=0)
ax.axvline(.5, color=INK, lw=1.0, ls=(0, (3, 3)), alpha=.8, zorder=1)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
ax.tick_params(axis="y", length=0)
ax.set_xlim(.42, 1.01); ax.set_xticks([.5, .6, .7, .8, .9, 1.0])
ax.set_xlabel("ROC AUC")
ax.text(.5, 1.0, "chance", transform=ax.get_xaxis_transform(), ha="center",
        va="bottom", fontsize=9.5, color=INK, alpha=.85)

# group labels, coloured to match the points
ax.text(1.005, ys[0] + .45, "AlphaFold confidence", color=POS, fontsize=10,
        ha="right", va="bottom", style="italic")
ax.text(1.005, ys[3] + .45, "ESM likelihood", color=NEG, fontsize=10,
        ha="right", va="bottom", style="italic")

fig.savefig(os.path.join(OUT, "fig-confidence.svg"))
for name, bg in (("light", "#ffffff"), ("dark", "#1b1a19")):
    fig.savefig(os.path.join(OUT, f"fig-confidence-check-{name}.png"), dpi=150,
                transparent=False, facecolor=bg)

print(f"disorder AUC {auc_disorder:.3f} over {len(res)} residues")
print(pg.groupby("coarse_selection_type")[PGCOL].agg(["count", "median"]).round(3))
