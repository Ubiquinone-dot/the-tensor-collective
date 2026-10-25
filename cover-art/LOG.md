# Cover art log

One entry per cover: which post it's for, the starting image it was painted from, the prompt, and where the result lives. Starting images go in `sources/`, named after the post (e.g. `sources/democracy-alignment.jpg`).

Covers are made in ChatGPT (DALL·E 3 for the older ones). See `style-pack/` for the style guide to drop into a new session.

**To add a cover:** save the starting image to `sources/<post-slug>.<ext>`, copy the template at the bottom, and fill it in.

---

## Democracy as an Alignment Problem

- **Post:** `site/content/posts/democracy-alignment/`
- **Cover:** `cover.webp` (1672×941), dark-mode `cover-dark.webp` (1672×941)
- **Date:** 2026-10-24
- **Tool:** ChatGPT
- **Starting image:** a classical painting of an orator addressing the Athenian assembly with the Acropolis behind. Likely Philipp von Foltz, *Pericles' Funeral Oration* (1852). *To confirm, and add to `sources/`.*
- **Prompt:** "Paint me an impressionist oil painting version of this painting pasted in the style of an exploding nebula." The dark cover was a night-time repaint of the day cover.
- **Chat:** https://chatgpt.com/share/6add4043-003c-83e8-b361-bcb8ec73d4b3
- **Notes:** the first cover was swapped for the current one on 2026-10-24.

## Why work on bio ML

- **Post:** `site/content/posts/why-bio-ml/`
- **Cover:** `cover.webp` (1024×1024). Same image as `site/static/imgs/DALL·E 2025-01-09 18.06.23 - An impressionist oil painting of a blank circle….webp`
- **Date:** 2025-01-09 (generated), reused 2026-10
- **Tool:** DALL·E 3
- **Starting image:** none, text-only prompt.
- **Prompt (from the filename, truncated):** "An impressionist oil painting of a blank circle at the center, surrounded by broad, sweeping strokes of vibrant colors, with a focus on deep green ton[es]…"

## On building diffusion models (RFD3)

- **Post:** `site/content/posts/rfd3/`
- **Cover:** `site/static/imgs/dna_binder.png` (1536×1024). An "exploding" version also exists: `site/static/imgs/dna_binder_exploding.png`.
- **Date:** 2025-09-20
- **Tool:** ChatGPT
- **Starting image:** *unknown.* Probably a render of a designed DNA-binding protein. *Add to `sources/`.*
- **Prompt:** alt text says "an impressionist painting painted in the style of an exploding nebula".

## From Matrices to Transformers

- **Post:** `site/content/posts/transformers/`
- **Cover:** `site/static/imgs/man2.png` (2000×2000)
- **Date:** 2023-12-07
- **Tool:** DALL·E 3
- **Starting image:** none, text-only prompt.
- **Prompt:** alt text says "An impressionist oil painting of a man, painted in the style of an exploding nebula".

## Introduction

- **Post:** `site/content/posts/Intro.md`
- **Cover:** `site/static/imgs/man.png` (2000×2000)
- **Date:** 2023-12-07
- **Tool:** DALL·E 3
- **Starting image:** none, text-only prompt.
- **Prompt:** caption says "An impressionist oil painting from DALLE-3 of a man gazing into an exploding nebula."
- **Dark-mode cover:** `site/static/imgs/man-dark.webp` (1254×1254), added 2026-10-25. Made in ChatGPT as a night-time repaint of `man.png`.

## Unused

- `site/static/imgs/man3.png`: alternate DALL·E 3 portrait from the same 2023 batch.
- `site/static/imgs/bug.png`, `DALLE3Bug.png`: ladybird under an exploding nebula, two variants (2023).

---

## Template

```markdown
## <Post title>

- **Post:** `site/content/posts/<slug>/`
- **Cover:** `cover.webp` (<w>×<h>)
- **Date:** YYYY-MM-DD
- **Tool:** ChatGPT
- **Starting image:** `sources/<slug>.<ext>`, <what it is and where it came from>
- **Prompt:** "<prompt>"
- **Chat:** <share link>
- **Notes:**
```
