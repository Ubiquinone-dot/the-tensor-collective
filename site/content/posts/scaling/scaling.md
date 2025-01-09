---
title: "Efficiency is good, but scale is better"
date: 2025-01-09T01:13+00:00
author: ["Jasper Butcher"]

searchHidden: false
ShowToc: true
TocOpen: false

categories: []
tags: [2025]
description: "I've begun to notice a trend in recent successful architectures across protein modelling and LLMs. In this article, I discuss how very impressive models have been designed by focusing on scale, rather than efficiency and in some cases requiring unintuitive choices to do so."

cover:
  image: imgs/DALL·E 2025-01-09 18.06.23 - An impressionist oil painting of a blank circle at the center, surrounded by broad, sweeping strokes of vibrant colors, with a focus on deep green ton.webp
  alt: 

weight: 10
draft: false

---

*Jan 9 2025*

### Introduction
I recently came across a very interesting paper by Meta, [Pagnoni 2024 et al.](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/). The title of which, *Byte Latent Transformer: Patches Scale Better
Than Tokens*, has a rather interesting assumption; it may be more desirable to scale better than to simply perform better.

The main result of their paper is unsurprisingly that their new transformer, BLT, scales better than previous SOTA techniques:

![[Pasted image 20250109011752.png]]
*Figure 1 [Pagnoni 2024 et al.](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/).*

This is really a remarkable result, because they train up to 8B parameters, and if the scaling trend were to continue up to, say, 40B (the size of the LLAMA-3 model which is said to be on par with GPT-4), then their technique would very likely unravel a better model.

The central idea behind their proposed technique is to change the tokenizer strategy itself to be learnt, and thereby allow the model to dynamically allocate resources during inference. There are only few added parameters, and the architecture scales much better 

In this blog post, I argue that their results are similar to what we've seen across the board in various domains, where large labs have seen significant benefit from pivoting to more scalable architectures, rather than efficient ones.
