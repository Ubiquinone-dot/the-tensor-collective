# Collected Tensors cover art: style guide

Paste this file (or attach it) at the start of a new ChatGPT image session, along with the images in `references/`. Then paste the starting image for the new post and ask for the cover.

## What I'm making

Cover images for my blog, *Collected Tensors* (collectedtensors.com), essays on protein design, ML and AI. Every cover is a painted reinterpretation of a starting image: a classical painting, a structure render, a photo or a figure from the essay.

## The house style

- **Medium:** impressionist oil painting. Thick, visible impasto brushstrokes, broken colour, painterly texture all over the canvas. Never photographic, never smooth digital art, never flat vector.
- **The signature motif:** an *exploding nebula*. Strokes radiate outward from a bright core, like a supernova or a starburst, with flecks of stars in the dark areas. The subject either sits in front of the nebula, is lit by it, or is partly dissolving into it.
- **Palette:** deep emerald and forest greens, crimson and burnt orange, cobalt and midnight blue, with warm cream or white-gold at the light source. Saturated, but rich rather than neon.
- **Composition:** one clear subject that still reads at thumbnail size. Keep the composition of the starting image recognisable; the style transfer should not invent a new scene.
- **No text,** no letters, no logos, no signatures, no borders or frames.

## The prompt I usually use

> Paint me an impressionist oil painting version of this painting pasted in the style of an exploding nebula.

That's usually all it needs. Useful add-ons when the first attempt is off:

- *"Keep the composition and figures of the original, only change the painting style."*
- *"Make the brushstrokes thicker and more visible, like heavy impasto."*
- *"Push the palette toward deep greens and crimson."*
- *"Make the nebula burst from behind the subject, radiating outward."*

## Formats

- **Aspect:** landscape about 16:9 (e.g. 1672×941) or 3:2 (1536×1024) works best in the post header. Square (1024×1024) is fine for shorter notes.
- **Export:** I convert to `.webp` and save it as `cover.webp` in the post's folder.

## Dark-mode variant

Some posts also have a night-time cover that the site crossfades to in dark mode (`cover-dark.webp`). To make one, give back the finished day cover and ask:

> Repaint this same painting at night: same composition and brushwork, a dark midnight-blue sky full of stars and nebula light, the figures lit by the glow.

## The references

`references/` holds finished covers from the blog. Treat them as the target look:

| File | Post |
|---|---|
| `democracy-day.jpg` / `democracy-night.jpg` | Democracy as an Alignment Problem (day and dark-mode covers) |
| `why-bio-ml.jpg` | Why work on bio ML |
| `dna-binder-exploding.jpg` | On building diffusion models (RFD3) |
| `bug.jpg` | Early DALL·E 3 experiment |
| `man-gazing.jpg` / `man-gazing-night.jpg` | Introduction post (day and dark-mode covers) |
