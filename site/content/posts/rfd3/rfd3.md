---
title: "On building diffusion models"
date: 2025-09-19T14:37+00:00
author: ["Jasper Butcher"]

searchHidden: false
ShowToc: true
TocOpen: false

categories: ["Diffusion", "Protein Design"]
tags: ["Protein design", "Enzyme Design"]
description: "With this article, I wanted to share to share some of the experiences we had building RFD3, and how this project changed the way I thought about building generative models to solve real world problems."

cover:
  image: imgs/dna_binder.png
  alt: "Generated image of a DNA binding protein, an impressionist painting painted in the style of an exploding nebula"

weight: 9
draft: false

---
BioRXiv DOI [https://doi.org/10.1101/2025.08.14.670328](https://www.biorxiv.org/content/10.1101/2025.09.18.676967v1) | Summary on [X](https://x.com/butcher_jasper/status/1968900323071713320)

## Introduction
As of the time of writing, we've just released RFdiffusion3's preprint. A component I often feel is missing from papers is the journey that went on behind the scenes. I went into the field of building models 3 years ago completely blind, and would've loved to know what it looks like on the inside. With this article, I wanted to share to share some of the experiences we had, and how this project changed the way I thought about building generative models to solve real world problems.

### Hallucination and my early attempts at active site scaffolding

C.A. 2022, I interned at the BakerLab doing a summer project on [hallucination for protein design](https://www.nature.com/articles/s41586-021-04184-w). Hallucination was birthed using the principles from [Google deep dream](https://research.google/blog/inceptionism-going-deeper-into-neural-networks/); creating vivid images using classification networks by "inverting" the network to amplify the signals in the input image. Since folding networks, such as trRosetta, could predict sidechain coordinates as well as backbone coordinates, one could amplify the signals of the output to improve input to the model (the sequence logits).

I became interested in the problem of active site scaffolding, being a chemist who only understands atoms and not complicated things like proteins. Active site scaffolding is a problem within which you have an enzyme's minimal atomic constraints (e.g. the precise coordinates of atoms interacting with the molecule to be catalysed) and you have to create a protein which has a geometry capable of satisfying those constraints.

The approach I worked on (figure **1**) would, at every iteration, insert the nearest inverse rotamer to the active site tipatoms. It would, in theory, allow you to start from a minimal specification of an active site without telling the model where the active site residues belong to, and it would figure out the optimal location throughout the trajectory.

{{< figure src="/posts/rfd3/inverse_rotamer_design.png" caption="Inverse rotamer design with hallucination diagram" >}}

This worked reasonably well, but was admittedly a hacky solution to a hard problem and hallucination was quickly superseeded by the development of RFdiffusion(1).

### What did RFD2 teach us about active site scaffolding?
RFdiffusion2 handles this concept much more elegantly, and formally introduced the concept of "unindexing". Unindexing, in machine learning terms, means you provide the identities and spatial relationships of the fixed atomic motif (active site), but you do not provide any information about where in the protein those atomic motifs should belong to.

`In image generation terms, it's equivalent to providing a context, like an image of a banana, say, and enforcing the reference is exactly copied into the generated image, without saying where in the image it should belong. It's "unindexed" - without position in the generated image`

This was a huge innovation in *de novo* protein design, especially enzyme design. It relaxes the amount of information you need to specify *a-priori*, and therefore enables significantly more diverse generation. It enables the model to use all of it's vast knowledge of the proteins it's seen to fill in the blanks.

One key limitaiton of RFD2 was that it had to use multiple diffusion tracks to unindex a tipatom. Diffusion was learnt through a backbone track $x_t$ and a sidechain track $s_t$ (attached to the tipatom conditions of known residue identity $c$) (figure **2**a). This meant, in order for $s_t$ to find it's index in $x_t$, two diffusive processed had to converge and join up to make the full all-atom structure; $x_t$ and $s_t$ (figure **2**b).

{{< figure src="/posts/rfd3/rfd2.png" caption="Diagram from Butcher et al. 2025, BioRXiv" >}}

### AlphaFold3 and the "fold" in RFdiffusion
Both RFD1 and RFD2 were forged from their respective protein folding architectures (RosetTTAFold and RF2AA). With RFD3 we built on the code for RF3 - consisting primarily of the AlphaFold3 architecture. AF3 introduced all-atom diffusion in a novel way, and did so without using **frames** or **equivariance**, two components which signifcantly increased the complexity of previous models.

Part of what inspired me from AF3 was concurrent work from Pagnoni *et al.* 2025 ("Byte Latent Transformer") on creating end-to-end trained LLMs without tokenizers. Pagnoni suggested that indeed the use of heavy inductive biases, such as efficient tokenizers, might not be necessary when you scale - similar to the work of AF3 wrt removing equivariance and frames. So perhaps efficiency is good, but scale is better.

Additionally, inspirations from their work also directly made it into the preprint version of RFD3 in the form of using cross attention between tracks, drawing on the similarity between the fields: bytes are to words as atoms are to residues. 
## Development of RFD3
We therefore shifted our focus from building super advanced models, to building clean, maintainable code with as little machine learning complexity as we could get away with - which worked in great harmony with the release of [AtomWorks](https://www.biorxiv.org/content/10.1101/2025.08.14.670328v1) and [ModelForge](https://github.com/RosettaCommons/modelforge).
### How to build on rough code
I think one of the things that enabled us to iterate the basics of the model fast were **building fast evaluation pipelines at every level of signal**. That includes:
- Minutes: easily monitoring the stability of the loss function
- Hours: whether the basics of structure are being learnt right (i.e evaluating the number of clashes and chainbreaks on the fly)
- Days: Whether the finer details of structure and sequence look good (rotamer quality, sequence distribution)
- Days/Weeks: shallow AlphaFold(2) evaluations on simple tasks like unconditional generation or motif scaffolding. 
- Weeks: deep evaluations, such as those presented in the paper. This undoubtedly was the hardest part, setting these up is a real challenge when you branch out to several tasks.
I think this is unfortunately where compute enables a few advantages; streamlining this process inefficiently is easier if you're with a big company, when you don't have unlimited resources, you have to find ways to make it efficient.

I'd say this is one of the places where a small, efficient model reaps a tonne of benefits. Scaling up and making the model better is easy, getting the right amount of signal is hard. [WandB](wandb.ai) is the greatest tool for this, hook it up to your workflows. Efficient automatic logging is a must, especially if you're compute constrained. You don't want to have to "test if the model looks good", the test results should be in your browser.
### Define your problem, and make it hard
Evaluations are an **extremely hard problem** in protein design, especially with the field moving so fast. Unfortunately, every benchmark has problems with  - It's basically like asking ChatGPT whether your essay sounds good, sometimes it'll hallucinate and lie to you (it certainly does with mine.)

What makes benchmarking *in silico* really hard is that pushing the boundaries of design, designing proteins previously not thought possible, by definition will be at the edge of what we can tell *in silico* to begin with - if things were as easy as passing a self-consistency check with AlphaFold3 we would've cured all cancer by now (although who knows maybe it is so for AlphaFold4!); model development has to be interweaved with working on hard, real-world, design problems.

Coming back to active site scaffolding, active site scaffolding is a hard problem, not because the *in silico* metrics are particularly good (they're alright as of 2025), but the problem is a geometric problem for the model and can therefore be well-quantified at the level of shallow signal without additional ML tools like ProteinMPNN or AF3. It's hard but well defined.
## Conclusions
RFD3 became easy to work with for enzyme design because it was no longer two diffusion tracks ($x_t$, $s_t$) that had to join up; it was simply one all-atom track $x_t^{\mathrm{all-atom}}$ that had to join up with it's condition $c$. This approach had minimal added complexity added to the architecture and training, and meant we could spend a lot of time on code cleanliness and allowed other people to collaborate together on one model. A second point to code, I think we got the principles of bridging metrics right; automatic shallow evaluations and lot's of them to look at; this meant when things weren't working (which they weren't quite often) we had metrics to evaluate how and could backtrack why - It was easier to send off the model to train on a new task than it was to evaluate whether it had learned anything, so we'd often do the former first and then start working the latter during training. Lastly, nothing beats working with talented and smart people, the team was unbelievable, it was incredibly fun the whole way through.

-Jasper,
20 Sep 2025

{{< figure src="/posts/rfd3/codiffusion.gif" caption="Co-diffusion of active site and ligand with RFdiffusion3." >}}
