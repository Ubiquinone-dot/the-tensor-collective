2023-11-17 22:16
Status: #idea #article #draft
Tags: [[The Tensor Collective]]

---
title: "Hallucination isn't a bug, it's a feature"
date: 2023-11-17T22:16+00:00
author: ["Jasper Butcher"]

searchHidden: false
ShowToc: true
TocOpen: false

categories: []
tags: []
description: ""

cover:
  image: PATH.png
  alt: DALLE-3 generated image with prompt - Impressionist oil painting illustrating a bug, but instead of its natural features, it bursts with vibrant colors and patterns evoking an exploding nebula

weight: 10
draft: false

---

_In credit of large language models_

With the advent of large language models (LLMs), much research and debate has naturally revolved around the flaws large language models exhibit. Ever wondered how ChatGPT manages to overlook aspects, but will immediately apologise and correct it's error once prompted? Hallucinations give fascinating insight into the cognition ongoing within a large language model.

Here, we'll make the case that the general user experience of LLMs do not give full credit to their capacities. Discussing ways to probe the true knowledge within these models. We'll also see what efforts are being made to combat hallucinations and force the models to reference sources in their responses.



## There's no AGI in LLM

If LLMs are so good at english, what happens if we ask them to avoid writing any english words?

```
#USER: Write me a paragraph that looks like english but contains no words in the english dictionary

#GPT-4: "Florble grinty moopler, zibbering quibbly in the glorpish sun. Bingle zorps flibbered,  brintly under the zorgish moon. Jibberish blorps, frinting and glabbering, zorped the quibbly flibber. Grinty blorpish, the moopler zorped, zibbering glorpishly in the brintly sun.
```

This is actually surprisingly good, a lot of them are completely non-english words. Note half are all either companies, places or on urban dictionary; so it's not like these words have never been uttered before but they aren't in the Oxford dictionary at least. Note though that it hasn't completely removed all English words; what we're looking for is complete gibberish and no recognisable words at all. 

It's not actually difficult to string together a series of words that vaguely resemble the structure of the English language. There are a few ways you could deem an answer appropriate: common motifs in words (like -ing, -er and -ish), the lengths of the sentences, the use of common letters (so not z perhaps), the use of binder words (?). So while the quality of the response is **subjective**, we can instantaneously tell whether it's **correct**; making it quite an easy task for us to do as humans.

Let's retry, making it clear that words like 'it', 'the' and 'and' aren't allowed:

```
#GPT-4: Zibber flibber zog, blibber zog zog. Zibber flibber zog, blibber zog zog. Zibber flibber zog, blibber zog zog. Zibber flibber zog, blibber zog zog. Zibber flibber zog, blibber zog zog. Zibber flibber zog, blibber zog zog. Zibber flibber zog, blibber zog zog.
```

This fixed the issue, but it seems like it's just repeating the first sentence it made?

The common theme here is what we're probing is the models' ability to think _out of distribution_. At a high level, large language models are trained on a large corpus of text data; where they extract complex statistical patterns in language and update their parameters accordingly. Language models are probabilistic in the sense that they create distributions across all the possible words, subwords or characters that are likely to come after what your prompt was. They are called *autoregressive* as they then take the most probable word they predicted and use that too to infer the next word in their response.

Figure?

This is a task specifically designed to be difficult for LLMs (it even says so itself!), as it requires the explicit inference of unlikely sequences of words, not known to its dataset. Of course, GPT-4 may have been trained on texts containing gibberish; articles or webpages actually containing some combination of the words, making it possible for it to complete this task. In some ways, since GPT-4 was trained to be correct all the time, it's sometimes difficult for it to be wrong.

When you use ChatGPT, the GPT model predict the most-likely next word. However you can actually vary the sampling distribution across the next token that the model uses, in OpenAI's playground API. Let's try asking the same question as above but with a temperature of 1.0 instead of 0:

```
#GPT-4: Flibber goshtop zilber jibnosh, crango huplip twads zinxle. Sproggle zuntip, yalgrin bobble gupshoe floxil. Zebnork quaflit vungnip, splingo wognip rindle yublop. Ghiblon tufnip wablon, dwindle zoplit yibber quopxin. Jindle blop gixnip, flob wipsin yipble zuntop. I'm sorry if I misunderstood, but it's quite challenging.
```

Much better! Here we get varied-length sentences, quite a few different words, and an apology for some reason.

There's a relation here to google search I think is worth stating; google by it's functionality cannot hallucinate articles or websites. If you ask for something that isn't on the internet, it won't return you anything. The role that LLMs have started to fill is 

Equation?
Figure of T-dependent softmax?

## Hallucinations

Note this was done using the instruction-tuned GPT-4, so the answers may differ from the regular ChatGPT interface.
What we've gathered so far is that ChatGPT has *no knowledge of right and wrong*. It doesn't literally know things. If it did, it would of course had seen readily that it was incorrect in it's responses above. 

## On Solving Hallucinations



## Conclusion

The issue with hallucinations is, therefore, a feature of the transformer architecture. They're issues we will be able to fix, but not with bigger models or better training alone. They are, for now, an inherent issue with the way large language models attain and store information. Some day, it may be possible to inplant directly into the weights of a model specific facts, information or experiences, similar to how humans accumulate memories. The relationship between hallucinations and the concept of truth is nuanced and complex. The language models we use today are of course no better than an analogous google search, where we understand the result is not necessarily the truth but have been swayed to it's reliability with time. It's astonishing however, the level of computer cognition we have access to. The lack of understanding and control we have over the semantics of large language models reflects a general unease within the machine learning community; models are easy to train but hard to interpret. Efforts for interpretability in machine learning is a topic for itself, but here we've touched upon how the scale of the complexity of the issue of hallucinations is an incredibly complex one. Hallucinations serve as the inflection point for AGI, the point where statistical pattern matching becomes exact, robust semantic reasoning.
