---
title: "Efficiency is good, but scale is better"
date: 2025-01-09T01:13+00:00
author: ["Jasper Butcher"]

searchHidden: false
ShowToc: true
TocOpen: false

categories: []
tags: []
description: "I've begun to notice a trend in recent successful architectures across protein modelling and LLMs. In this article, I discuss how very impressive models have been designed by focusing on scale, rather than efficiency and in some cases requiring unintuitive choices to do so."

cover:
  image: imgs/DALL·E 2025-01-09 18.06.23 - An impressionist oil painting of a blank circle at the center, surrounded by broad, sweeping strokes of vibrant colors, with a focus on deep green ton.webp
  alt: 

weight: 10
draft: false

---

# Efficiency is good, but scale is better

### Introduction
I recently came across a very interesting paper by Meta, [Pagnoni 2024 et al.](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/). The title of which, *Byte Latent Transformer: Patches Scale Better
Than Tokens*, has a rather interesting assumption; it may be more desirable to scale better than to simply perform better.

The main result of their paper is unsurprisingly that their new transformer, BLT, scales better than previous SOTA techniques:
![[Pasted image 20250109011752.png]]Figure 1 [Pagnoni 2024 et al.](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/).

This is really a remarkable result, because they train up to 8B parameters, and if the scaling trend were to continue up to, say, 40B (the size of the LLAMA-3 model which is said to be on par with GPT-4), then their technique would very likely unravel a better model.

The central idea behind their proposed technique is to change the tokenizer strategy itself to be learnt, and thereby allow the model to dynamically allocate resources during inference. There are only few added parameters, and the architecture scales much better 

In this blog post, I argue that their results are similar to what we've seen across the board in various domains, where large labs have seen significant benefit from pivoting to more scalable architectures, rather than efficient ones.

### Equivariance or no equivariance?

[Wang et al. 2024](https://openreview.net/forum?id=XSwxy3bojg) spurred some debate in the conformational generation fields as to whether we should consider using non-equiviariant architectures for equivariant tasks. 

### Hard-priors and Soft-priors
The reason why BLT scales better than the previous techniques is simple. Constraining the model less to a certain vocabulary enables it to optimally learn this behaviour. 

This explanation I've heard before, and I think it's fascinating language. I mean why would behaviour implicit to the loss necessarily be *optimally learnt*? Why would the learned vocabulary be any better? I mean there's no extra loss term, the model is trained end-to-end as before.

The optimality of the unlocked features comes from the fact that you sculpt the loss landscape alongside it's functional space, let the ball of gradient descent roll and find it's optimal solution.
### The role of efficient architectures
There are scenarios where you cannot simply create big architectures. And those tend to be data-constrained domains.

AF2 distillation set as an example of efficient -> scalable transfer of information. Could you train AF3 without distillation? 

In other words, if you would like to incorporate pretraining with synthetic data using LLAMA-3, you wouldn't use BLT proposed in the paper, for a 2B sized model, you'd use the original architectures.

### Why scalable architectures will always fail
This new paradigm of scalable architectures really fascinates me. You're essentially giving up the steering wheel and saying 'I don't know how you're going to achieve this, but I'll see if you do'. Models do incredible things, be it discover the rotational- and translational symmetry of the data, or the full suite of impressive capabilities of large language models.

With scalable architectures you're asking for a bit more.

It's the same reason why in figure 

### What's next?
So, basically nothing scales better than MLPs right, simple seems to be a good heuristic for success.

### Conclusion

