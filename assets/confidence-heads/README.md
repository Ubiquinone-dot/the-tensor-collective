# Confidence heads track experiments nobody trained them on

Working note for the protein panels of the *Ladder of Biology* figure. Gathered
6 November 2026.

This directory is an archival asset outside `site/`. Netlify builds from `site/`, so
nothing here is published.

---

## The finding

A structure predictor is trained to place atoms, and to predict how wrong its own
atom placements are. A protein language model is trained to fill in a masked residue.
Neither objective contains a measurement. Yet the confidence each one emits
discriminates wet-lab outcomes across a wide spread of tasks, none of which appear in
either loss function.

The table is the argument. Every value is either recomputed from released data in
this directory, or taken from the cited paper.

| Task | What the head is | AUC | Source |
|---|---|---|---|
| Residue is intrinsically disordered | AF2 pLDDT | **0.933** | recomputed, [data](#1-disorder) |
| Two chains genuinely interact | AF2 confidence | 0.84 | [PPIscreenML](https://elifesciences.org/reviewed-preprints/98179) |
| Designed binder binds in vitro | AF2 ipTM | 0.636 | recomputed, [data](#5-designed-binders) |
| Missense variant is pathogenic | ESM1b likelihood | 0.905 | [Brandes et al. 2023](https://www.nature.com/articles/s41588-023-01465-0) |
| Deep mutational scan: stability | ESM2 650M | 0.790 (n=66) | [ProteinGym](https://github.com/OATML-Markslab/ProteinGym) |
| Deep mutational scan: activity | ESM2 650M | 0.761 (n=43) | [ProteinGym](https://github.com/OATML-Markslab/ProteinGym) |
| Deep mutational scan: organismal fitness | ESM2 650M | 0.748 (n=77) | [ProteinGym](https://github.com/OATML-Markslab/ProteinGym) |
| Deep mutational scan: expression | ESM2 650M | 0.742 (n=18) | [ProteinGym](https://github.com/OATML-Markslab/ProteinGym) |
| Deep mutational scan: binding affinity | ESM2 650M | 0.670 (n=13) | [ProteinGym](https://github.com/OATML-Markslab/ProteinGym) |

Deep-mutational-scan rows are medians over individual assays, and the figure draws
each assay as a faint point behind the median so the spread stays visible.

Two things worth noticing. The weakest row is binding affinity, which is also the
task the field most wants these scores for. And neither model wins everywhere: ESM
takes pathogenicity, AlphaFold takes disorder. Different objectives leak different
physics.

---

## Where each number comes from

### 1. Disorder

AlphaFold's per-residue confidence was trained to predict the lDDT of its own model
against the true structure, and never saw a disorder annotation. On the CAID3
Disorder-PDB set it separates curated disordered residues from ordered ones at
**AUC 0.933**.

| | |
|---|---|
| Proteins | 319 |
| Assessed residues | 99,239 (31,401 disordered, 67,838 ordered) |
| AUC, AF2 pLDDT | 0.933 |
| AUC, AF3 pLDDT | 0.932 |
| Median pLDDT, disordered | 44.6 |
| Median pLDDT, ordered | 94.1 |
| Ordered residues above pLDDT 70 | 93.3% |

Per protein, mean pLDDT against the fraction of residues curated disordered gives
Spearman ρ = −0.76 (n = 319, p ≈ 1 × 10⁻⁶⁰).

- Data and code: [BioComputingUP/af2_af3_idr_analysis](https://github.com/BioComputingUP/af2_af3_idr_analysis),
  the release accompanying that group's AlphaFold2-to-AlphaFold3 disorder comparison.
  Per-residue confidences are in
  [`data/disorder_prediction/AF2/AlphaFold2_data.tsv`](https://github.com/BioComputingUP/af2_af3_idr_analysis/blob/main/data/disorder_prediction/AF2/AlphaFold2_data.tsv);
  labels in
  [`data/disorder_prediction/references/disorder/disorder_pdb.fasta`](https://github.com/BioComputingUP/af2_af3_idr_analysis/blob/main/data/disorder_prediction/references/disorder/disorder_pdb.fasta).
- Labels originate from [DisProt](https://disprot.org) curation, routed through the
  [CAID challenge](https://caid.idpcentral.org/challenge). Curation draws on NMR,
  X-ray, SAXS and proteolysis, so the measurement is genuinely independent of the
  training signal, and CAID is a blind benchmark rather than the authors' own set.
- AF2 structures come from the [AlphaFold Database](https://alphafold.ebi.ac.uk);
  AF3 predictions were run on the AlphaFold Server and redistributed by those authors
  under the AlphaFold Server output terms.
- I recomputed the AUC from the per-residue files rather than quoting it. It matches
  the authors' own CAID-format evaluation (0.933 for AF2-pLDDT) in the same repo. The
  AlphaFold-disorder relative-solvent-accessibility variant scores higher still, 0.949.

### 2. ESM zero-shot across five assay types

[ProteinGym](https://github.com/OATML-Markslab/ProteinGym) v1.3, zero-shot
substitution benchmark, AUC metric, ESM2 650M. 217 deep mutational scans, grouped by
the benchmark's own coarse selection type. The model is trained purely on
masked-token prediction over UniRef; the labels are wet-lab measurements of
stability, enzyme activity, organismal fitness, expression level and binding affinity.

- Per-assay values:
  [`benchmarks/DMS_zero_shot/substitutions/AUC/DMS_substitutions_AUC_DMS_level.csv`](https://github.com/OATML-Markslab/ProteinGym/blob/main/benchmarks/DMS_zero_shot/substitutions/AUC/DMS_substitutions_AUC_DMS_level.csv)
- Model-level summary:
  [`.../Summary_performance_DMS_substitutions_AUC.csv`](https://github.com/OATML-Markslab/ProteinGym/blob/main/benchmarks/DMS_zero_shot/substitutions/AUC/Summary_performance_DMS_substitutions_AUC.csv)
- Category labels:
  [`reference_files/DMS_substitutions.csv`](https://github.com/OATML-Markslab/ProteinGym/blob/main/reference_files/DMS_substitutions.csv),
  column `coarse_selection_type`
- Licence: MIT.

Nothing is recomputed here. I joined the benchmark's published per-assay numbers to
the reference file's category labels and took medians;
[`data/proteingym_auc.csv`](data/proteingym_auc.csv) is that join, for four ESM
variants.

On binding affinity, the category Jasper asked about directly: median AUC 0.670 over
13 assays, the weakest of the five, with a wide spread. `SPIKE_SARS2_Starr_2020` sits
at 0.471, below chance; `DLG4_RAT_McLaughlin_2012` reaches 0.857.

The model-level summary also shows the effect is not confined to ESM. The same table
gives ESM2 650M an average AUC of 0.729 and ESM3 open 1.4B 0.755, against 0.741 for
EVE and 0.724 for the ESM-1v ensemble, so this is a property of the family rather than
one checkpoint.

### 3. Clinical pathogenicity

ESM1b scores all possible human missense variants zero-shot and reaches **ROC-AUC
0.905** on ClinVar, against 0.885 for EVE.

- Brandes et al., *Genome-wide prediction of disease variant effects with a deep
  protein language model*, **Nature Genetics** (2023).
  [doi:10.1038/s41588-023-01465-0](https://www.nature.com/articles/s41588-023-01465-0).
  Open-access copy: [PMC10484790](https://pmc.ncbi.nlm.nih.gov/articles/PMC10484790).
- The two figures above are quoted in the Results, p. 1514. The paper's Figure 2e
  compares 19 methods, but those values live in its Supplementary Table 2 rather than
  the text.

### 4. Interaction versus decoy — the softest number here

AF2's default confidence reaches roughly **AUC 0.84** at separating interacting pairs
from decoy pairings, against 0.88 for the trained PPIscreenML classifier.

- *PPIscreenML: structure-based screening for protein-protein interactions using
  AlphaFold*, eLife reviewed preprint
  [98179](https://elifesciences.org/reviewed-preprints/98179).
- Two caveats. I took 0.84 from the public reviewer summaries rather than the paper's
  own text, and the reviewers argue the decoy set may hold non-cognate or weak
  interactors rather than true non-interactors. Reviewer 1 also reports pDockQ at 0.71
  on this benchmark against 0.87 in its original paper, which shows how much the
  benchmark choice moves these numbers.
- If this row survives into the figure, replace it with a value read from the
  published figure. If it does not, dropping it costs the argument little.

### 5. Designed binders

Recomputed from the Adaptyv EGFR binder competition, round 2: 378 designs carrying a
bio-layer interferometry binding call, 53 of which bound.

| Score | AUC |
|---|---|
| Binder pLDDT | 0.656 |
| AF2 ipTM | 0.636 |
| Interface PAE | 0.612 |
| ESM2 pseudo-log-likelihood | 0.547 |

- [adaptyvbio/egfr_competition_2](https://github.com/adaptyvbio/egfr_competition_2),
  file [`results/result_summary.csv`](https://github.com/adaptyvbio/egfr_competition_2/blob/main/results/result_summary.csv).
  Round 1 is at [adaptyvbio/egfr_competition_1](https://github.com/adaptyvbio/egfr_competition_1).
- Licence: ODbL for the data, Apache 2.0 for the code.
- The assay is multi-cycle kinetics on BLI, run in duplicate, fitted 1:1, with
  equilibrium analysis where the kinetics outran the instrument.

A much larger study agrees on direction and does better with a sharper metric. Across
**3,766 experimentally characterised binders over 15 targets**, of which 436 (11.6%)
bound in vitro, AlphaFold3's `ipSAE_min` was the best single predictor and gave a
**1.4-fold increase in average precision** over the AF2 initial-guess interface PAE
the field had been using. Maximum-F1 threshold at `ipSAE_min` > 0.61.

- *Predicting experimental success in de novo binder design: a meta-analysis of 3,766
  experimentally characterised binders*, bioRxiv
  [2025.08.14.670059](https://www.biorxiv.org/content/10.1101/2025.08.14.670059).
- Data: [Zenodo 10.5281/zenodo.15722219](https://zenodo.org/records/15722219).
  Code: [DigBioLab/de_novo_binder_scoring](https://github.com/DigBioLab/de_novo_binder_scoring), MIT.
- I cited rather than replotted this one. The Zenodo deposit carries the compiled
  feature table if a success-rate-against-confidence curve is ever wanted.

---

## The caveat to keep in the prose

Confidence tracks whether a region has a structure at all, superbly. It is not a
calibrated dynamics measurement. Within folded regions, pLDDT does not reproduce the
fine ordering of NMR order parameters — *Gradations in protein dynamics captured by
experimental NMR are not well represented by AlphaFold2 models and other computational
metrics*, bioRxiv [2024.07.17.603933](https://www.biorxiv.org/content/10.1101/2024.07.17.603933).

So the claim to make is the strong, defensible one: these heads recover properties of
the molecule that were never in their loss. Not that they measure flexibility.

---

## Rebuilding the figure

```sh
cd assets/confidence-heads
python3 confidence.py        # -> out/fig-confidence.svg + light/dark check PNGs
```

Needs `numpy`, `pandas`, `matplotlib` and `scikit-learn`. `fig-confidence.svg` in
this directory is the committed render.

| File | What it holds |
|---|---|
| `confidence.py` | builds both panels |
| `data/AF2_res.csv.gz`, `data/AF3_res.csv.gz` | per-residue pLDDT and experimental disorder label |
| `data/proteingym_auc.csv` | per-assay AUC and category, four ESM variants |
| `data/plddt_calibration.csv` | fraction disordered per pLDDT bin |
| `data/per_protein.csv` | per-protein means behind the Spearman figure |

Colours follow the rest of the figure: ink `#8a8378`, teal `#1c9a85`, orange
`#d9673f`. Transparent background and no opaque label boxes, so it reads in both light
and dark mode.
