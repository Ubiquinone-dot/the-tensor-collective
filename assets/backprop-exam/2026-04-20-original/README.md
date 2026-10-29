# Original backpropagation exam and plot sources

Recovered from commit `776c315c41b8695b8357c98bfe11a71b2096bee9` of the original midterm repository, committed April 20, 2026 at 23:02 PDT. This is the 47-point version before commit `45af3b1` ("Trim and structure midterm Q1: 47 -> 22 pts"). Question 1.1 retains decomposition, graph drawing, numerical backpropagation with W=2 and b=3, and the analytical derivative (11 points total).

This directory is an archival asset outside `site/`. Netlify builds from `site/`, so these files are not included in the site's published contents.

## Read and edit

- [Questions](backprop.pdf)
- [Questions with solutions](backprop_soln.pdf)
- [Editable Typst source](question.typst)
- [Historical design notes](DESIGN_NOTES.md) (originally named CLAUDE.md)

The original PDFs, figure PDFs, and Python scripts are preserved byte-for-byte from the source commit. `question.pdf` is an additional historical PDF; use `backprop.pdf` and `backprop_soln.pdf` as the paired question/solution documents. `notes.txt` is an unfinished earlier training-code sketch, not part of the build. Some script comments retain older question numbering; the mapping below follows `question.typst`.

## Environment

Use Python 3.12 and Typst 0.14.2. The Python dependency versions in `requirements.txt` reflect the available reconstruction environment, not an original lockfile. The document pins the Typst package `@preview/cetz:0.4.2`; the first compile needs internet access unless that package and its dependencies are already cached.

```sh
cd assets/backprop-exam/2026-04-20-original
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install Typst separately and ensure `typst` is on your PATH. The document uses New Computer Modern. The data is generated in the scripts using fixed seeds; no external dataset is required.

## Rebuild

Run commands from this directory. They overwrite the corresponding generated PDFs; retain a copy if you want to compare with the archived originals.

```sh
make all        # regenerate all figures used in the exam, then both PDFs
make bce-plots  # optional binary-cross-entropy comparison figures
```

To compile only the documents using the checked-in figures:

```sh
make pdfs
```

Equivalent document commands:

```sh
typst compile --input answers=false question.typst backprop.pdf
typst compile question.typst backprop_soln.pdf
```

## Plot-generation map

| Command | Generated files | Use |
| --- | --- | --- |
| `python plot_sigmoid_deriv.py` | `plot_1_4.pdf` | 1.2: sigmoid derivative |
| `python train.py` | `plot_moons.pdf`, `plot_1_5.pdf` | 1.4: moons data and four initialization regimes |
| `python train_ln.py` | `plot_ln.pdf` | 1.5: LayerNorm loss curves and gradient comparison |
| `python plot_boundary.py` | `plot_boundary.pdf` | 1.6: MLP and KNN decision boundaries |
| `python train_bce.py` | `plot_1_5_bce.pdf` | Optional BCE initialization comparison |
| `python train_ln_bce.py` | `plot_ln_bce.pdf` | Optional BCE LayerNorm comparison |

`make` selects Matplotlib's noninteractive Agg backend. For direct script execution on a headless machine, set `MPLBACKEND=Agg`. All scripts are self-contained; PyTorch is used by the LayerNorm scripts. The decision-boundary script trains for 30,000 epochs and evaluates KNN on a 300 x 300 grid. Plot filenames reflect earlier numbering and are retained for source compatibility. Regenerated files may differ in PDF metadata, fonts, or floating-point details across environments.

## Validation

All six plot scripts ran successfully with Python 3.12.12 and the pinned dependency versions. Both PDF variants compiled successfully with Typst 0.14.2 using freshly generated figures. These checks ran in a scratch directory so the committed historical PDFs remain unchanged.
