# Midterm exam-question styling notes

Reference for future exam-question work in Typst. Captures the Typst
setup, the question-layout conventions, and the narrative arc that
emerged from iterating on this backprop question.

## Typst workflow

- Two PDFs from one source via `sys.inputs`:
  - `typst compile question.typst backprop_soln.pdf` — with answers
  - `typst compile --input answers=false question.typst backprop.pdf` — question sheet only
- Toggle helper at top of file:
  ```typst
  #let show_answers = sys.inputs.at("answers", default: "true") == "true"
  #let ans(body) = if show_answers {
    block(text(fill: rgb("#143c8c"), style: "italic")[
      #text(style: "normal", weight: "bold")[Answer.] #body
    ])
  }
  ```
- Figure numbering: `#set figure(supplement: [Fig.])` with a caption show-rule
  that renders only `*Fig. N*` (no caption body). Figures referenced via
  `@fig-name` where each `#figure` has a `<fig-name>` label.
- Use `#pagebreak(weak: true)` between top-level questions.
- Font: New Computer Modern, 10pt, `par(leading: 0.65em, justify: false)`.
- `cetz:0.4.2` for computation-graph diagrams (not 0.3.x — coord-resolution
  incompatibility with Typst 0.14).
- Raw Python code uses triple-backtick `python` fences directly in Typst.
- Create a local uv venv in the project dir for the plotting scripts:
  `uv venv && source .venv/bin/activate && uv pip install numpy matplotlib seaborn torch`.

## Paper colours (from the global style rules)

```
paper_teal    #4FB9AF   ← primary
paper_pink    #FFACB7
paper_indigo  #08415C
paper_coolgrey #9596C6
```

All plots use `sns.set_style("ticks")` + `sns.set_style("white")`,
`rcParams['font.family'] = 'serif'`, `dpi=300`, figsize ≈ `(4, 3)` by
default (wider for multi-panel). Use `plt.sca(axs[i])` style (not
`ax.plot(...)`).

## Question layout conventions

- Top-level header: `*1.N Short title.*` followed by intro prose. Bold
  number, short sentence-case title, period.
- Plots belong **at the top** of the question that introduces them, with
  intro text wrapping them. Subparts go below.
- Subparts: `*(a)*`, `*(b)*`, … with point tag right-aligned via
  `#h(1fr) (N pts)` at the end of each subpart statement.
- Each question has a single `#ans[...]` block after all subparts; the
  block mirrors the subpart labels inside: `*(a)*`, `*(b)*`, …
- Atomic questions (no subparts) get one overall `(N pts)` tag on the
  question line.
- Boldface convention: matrices use `bold(W)_1, bold(W)_2`; scalars and
  bias vectors stay non-bold. Call the convention out explicitly the
  first time it matters (e.g. "boldface to distinguish them from the
  scalar $W_1, W_2$ of 1.3").
- Inline figure refs use "@fig-X" and assume no caption text — the
  reference reads "Fig. N" inline.

## Answer-style rules

- **Punchy, not essay.** One to three sentences per subpart in most
  cases. Numerical parts carry the arithmetic inline.
- Every claim traces back to a prior derivation or a figure
  ("1.2(b)", "@fig-sigprime", "1.4B").
- **Flag own doubts.** Do not overclaim; prefer "more robust than" over
  "strictly better than" when the evidence is equivocal (e.g. LN vs.
  well-tuned init). The user has pushed back repeatedly on oversold
  narratives.
- No emojis.

## Narrative arc (this exam)

Scale deliberately from scalar primitives up to matrix networks, with
each new section reducing to the previous when its new feature is
switched off.

1. **1.1 Scalar sigmoid unit.** Decompose $f = \sigma(Wx+b)$ into
   primitive ops, draw the graph, compute one numerical forward +
   backward. Derive $\partial f / \partial W$ analytically. No plot.
2. **1.2 Sigmoid-derivative curve.** Introduce `plot_1_4.pdf` as the
   graph of the formula students just derived. Explain shape, chain
   to $L = (f-y)^2$, note the $(f-y) = 0$ perfect-model case, state
   the update rule and motivate dead-gradient intuition.
3. **1.3 Two-layer scalar composition.** Graph, numerical pass
   (values chosen to give a clean ~5× gradient gap between $W_1$ and
   $W_2$ — not a dramatic 40× so the narrative stays honest),
   explanation of the extra $\sigma'$ factor.
4. **1.4 FCN + init regimes.** Promote to bold matrices, define the
   highlighted "single stream" as the 1.3 scalar chain when off-path
   entries are zero. Train on two-moons under four inits (zero,
   Gaussian, huge, symmetric) — the four fail in distinct ways that
   factor as (dead-gradient) × (symmetry preserved). Ask for one
   alternative loss; BCE is the canonical answer.
5. **1.5 LayerNorm.** Insert LN before the hidden sigmoid. Ask for the
   single-stream Jacobian (just $1/s$ at single-stream granularity —
   the canonical clean answer, with cross-unit coupling noted as a
   parenthetical). Centralization + robustness framed as one question.
   Two figure-interpretation questions follow (faster convergence,
   larger gradient at both layers).
6. **1.6 KNN contrast.** Non-parametric comparison. Python
   fill-in-the-blanks on `knn_predict`, complexity, why no gradient
   descent, decision-boundary figure with four anonymized panels
   (MLP + K ∈ {1, 5, 200}), K=N ⇒ constant-predictor. Class balance
   ($N_0=106, N_1=94$) stated before the plot so students can explain
   the K=200 degenerate case quantitatively.

## Point-allocation discipline

- Atomic questions ≈ 1–2 pts per primitive op or gradient node.
- Numerical forward+backward passes: ~4–6 pts (one point per
  gradient node computed).
- Conceptual sub-parts: 1 pt each; 2–3 pts if explicitly asking for
  two things or deeper justification.
- Code fill-in-the-blanks: 1 pt per blank.
- Overall budget was **47 pts**; not a sacred number — the user said
  "just make it fair" over any total target.

## Cross-reference discipline

- When re-numbering sections, grep for every old number in the answer
  text too — old 1.4(c) became 1.2(b); old 1.5 became 1.3; etc. Every
  `(1.Xy)` tag in an answer needs to be re-pointed.
- "The 1.X stream" = the scalar chain from §1.X as a conceptual referent
  that returns in later sections.

## Plots/scripts in the repo

- `plot_sigmoid_deriv.py` → `plot_1_4.pdf`: bare $\sigma(W)(1-\sigma(W))$ curve,
  no y-ticks, no annotations.
- `train.py` → `plot_moons.pdf` (data scatter) + `plot_1_5.pdf` (4-panel
  loss curves A–D). NumPy-only MLP, MSE loss, lr=0.01, 5000 epochs.
- `train_ln.py` → `plot_ln.pdf`: 2-panel, left = loss curves for LN vs
  no-LN under large init; right = per-entry gradient magnitude at $W_1$
  vs $W_2$ at epoch 1000 (box + scatter, log y-axis). PyTorch.
- `plot_boundary.py` → `plot_boundary.pdf`: 1×4 decision-boundary strip.
  MLP was retrained at lr=1.0, 30000 epochs to fit well under MSE
  (lr=0.01/5000 epochs undertrains visibly due to output-sigmoid
  saturation slowdown). Legend at bottom shows prediction-region vs.
  training-point semantics. Panels shuffled (A=K=5, B=MLP, C=K=1,
  D=K=200) so students reason from shape, not position.
- `train_bce.py`, `train_ln_bce.py`: BCE variants explored during design.
  BCE preserves 1.4 story but collapses the 1.5 LN rescue — MSE is the
  pedagogically correct choice for the LN narrative.

## Things that broke during iteration (avoid next time)

- cetz 0.3.x panics on coord resolution in Typst 0.14 — pin 0.4.2.
- `#sim` isn't a Typst symbol; use `$tilde.op$`.
- Per-unit-variance Gaussian init on 4 hidden sigmoids + MSE + lr=0.01
  undertrains the MLP visibly; `plot_boundary.py` uses lr=1.0 and 30000
  epochs to produce a boundary that actually separates the moons.
- Class balance 100/100 at K=N gives a tied degenerate KNN; the
  shuffled moons dataset ended up 106/94 which is why the K=200 panel
  resolves cleanly to constant class-0.
