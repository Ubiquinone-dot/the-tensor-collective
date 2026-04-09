---
title: "LaTeX and bibliography demo"
date: 2026-04-09T12:00:00+00:00
author: ["Jasper Butcher"]
draft: false
description: "A short post verifying LaTeX rendering and .bib-driven references work end-to-end."
weight: 8
---

This post is a smoke test for two things:

1. Inline and display LaTeX via MathJax.
2. Citations pulled from a `.bib` file with no manual formatting.

## Inline and display math

The forward diffusion process adds Gaussian noise over $T$ timesteps:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{1-\beta_t}\,x_{t-1},\, \beta_t \mathbf{I}\right).
$$

With the standard reparameterisation $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ we can sample directly from the data:

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar{\alpha}_t}\,x_0,\, (1-\bar{\alpha}_t)\mathbf{I}\right).
$$

Inline works too: the noise prediction loss $\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\,\|\epsilon - \epsilon_\theta(x_t, t)\|^2\,\right]$ is the standard training objective.

## Citations

Denoising diffusion probabilistic models {{< cite "ho2020denoising" >}} are the foundation of modern score-based generative modelling. They've since been adapted to protein structure generation, with {{< cite "watson2023rfdiffusion" >}} being one of the most successful examples in the field. Equivariant variants were proposed earlier {{< cite "anand2022protein" >}}.

You can also chain keys: {{< cite "ho2020denoising,watson2023rfdiffusion" >}}.

{{< bibliography >}}
