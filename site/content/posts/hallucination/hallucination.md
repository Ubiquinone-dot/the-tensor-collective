
---
title: "Hallucination isn't a bug, it's a feature"
date: 2023-11-17T22:16+00:00
author: ["Jasper Butcher"]

searchHidden: false
ShowToc: false
TocOpen: false

categories: [ "NLP" ]
tags: [ "GPT-4", "Transformers" ]
description: "Insight into the cognition behind large language models with a simple conversation with GPT-4."

cover:
  image: imgs/DALLE3Bug.png
  alt: DALLE-3 generated image with prompt - Impressionist oil painting illustrating a bug, but instead of its natural features, it bursts with vibrant colors and patterns evoking an exploding nebula

weight: 10
draft: false
---

*Simple insights into the cognition behind large language models.*

With the advent of large language models (LLMs), much research and debate has naturally revolved around the flaws large language models exhibit. Ever wondered how ChatGPT manages to overlook aspects, but will immediately apologise and correct it's error once prompted? Hallucinations give fascinating insight into the cognition ongoing within a large language model.

Here, we'll make the case that the general user experience of LLMs do not give full credit to their capacities. Discussing ways to probe the true knowledge within these models. We'll also see what efforts are being made to combat hallucinations and force the models to reference sources in their responses.

## There's no AGI in LLM
### Language models don't speak gibberish

If LLMs are so good at english, what happens if we ask them to avoid writing any english words?

```bash
#USER: Write me a paragraph that looks like english but contains no words in the english dictionary
#GPT-4: "Florble grinty moopler, zibbering quibbly in the glorpish sun. Bingle zorps flibbered,  brintly under the zorgish moon. Jibberish blorps, frinting and glabbering, zorped the quibbly flibber. Grinty blorpish, the moopler zorped, zibbering glorpishly in the brintly sun.
```

This is actually surprisingly good, a lot of them are completely non-english words. Note half are all either companies, places or on urban dictionary; so it's not like these words have never been uttered before but they aren't in the Oxford dictionary at least. Note though that it hasn't completely removed all English words; what we're looking for is complete gibberish and no recognisable words at all. 

It's not actually difficult to string together a series of words that vaguely resemble the structure of the English language. There are a few ways you could deem an answer appropriate: common motifs in words (like -ing, -er and -ish), the lengths of the sentences, the use of common letters (so not z perhaps). So while the quality of the response is **subjective**, we can instantaneously tell whether it's **correct**; making it quite an easy task for us to do as humans.

Let's retry, making it clear that words like 'it', 'the' and 'and' aren't allowed:

```bash
# GPT-4: Zibber flibber zog, blibber zog zog. Zibber flibber zog, blibber zog zog. Zibber flibber zog, blibber zog zog. Zibber flibber zog, blibber zog zog. Zibber flibber zog, blibber zog zog. Zibber flibber zog, blibber zog zog. Zibber flibber zog, blibber zog zog.
```

This fixed the issue, but it seems like it's just repeating the first sentence it made?

At a high level, large language models are trained on a large corpus of text data; they extract complex statistical patterns from language and update their parameters accordingly. They are probabilistic in the sense that they create distributions across all the possible words, subwords or characters that are likely to come after what your prompt said. They are called *autoregressive* as they then take the most probable word they predicted and use that too to infer the next word in their response.

The common theme here is what we're probing is the models' ability to think out of it's knowledge distribution. This is a task specifically designed to be difficult for LLMs, as it requires the explicit inference of unlikely sequences of words, not known to its dataset. Of course, GPT-4 may have been trained on texts containing gibberish; articles or webpages actually containing some combination of these words, making it possible for it to complete this task. In some ways, since GPT-4 was trained to be correct all the time, it's sometimes difficult for it to be wrong.

So why does GPT-4 repeat itself? One way to think about this would be to see that the first word is the "hardest" for the model to generate; it hasn't had many samples in it's database for this. Once it has created the first, it may be recognising it's own output as a correct answer and therefore begins a chain of similar words; `Zibber flibber zog` are somewhat similar in their phrasing.

Importantly it's not that the sampling process is intelligently done by the model. It is the "modelling" process that we consider intelligent. It doesn't know that the task is to find less probable words and therefore should essentially flip it's model distribution upside down to acomplish the task, the only thing it "knows" is how to create a distribution over possible answers. 

This is the origin of why some users of ChatGPT report well-structured, yet vague and "average" answers to their questions. LLMs reason in a way that is probabilistic and holistic, yet uniform and completely disconnected from making semantic links.

### The temperature parameter
This sections is a bit of an aside for the sake of curiosity; When you use ChatGPT, the GPT model predict the most-likely next word. However you can actually vary the sampling distribution across the next token that the model uses, in OpenAI's playground API. Let's try asking the same question as above but with a temperature of 1.0 instead of 0:

```bash
# GPT-4: Flibber goshtop zilber jibnosh, crango huplip twads zinxle. Sproggle zuntip, yalgrin bobble gupshoe floxil. Zebnork quaflit vungnip, splingo wognip rindle yublop. Ghiblon tufnip wablon, dwindle zoplit yibber quopxin. Jindle blop gixnip, flob wipsin yipble zuntop. I'm sorry if I misunderstood, but it's quite challenging.
```

Much better! Here we get varied-length sentences, quite a few different words and it even apologises for the how hard it found the task?

The temperature parameter allows less likely words to be sampled. In the limit of high temperature we'd expect the answer to be gibberish anyway, but it is intersting to see exactly where the turning point is. My few tests seemed to indicate it was only at a temperature of 1 where the model managed to get "out" of the loop of repeating itself.

The exact parameterisation for the GPT-4 playground isn't of course public information but it's very likely to be related to the following:
$$
P(word_i) = Softmax(z_i \ / \ T) = \frac{e^{z_i/T}}{\sum_{j} e^{z_j/T}}
$$
The $T$ here is the temperature parameter. The output of the model will be some distribution across parameters, $z_i$. The softmax function converts these into probabilities which we can sample from, giving us the next word in the sequence. The figure below shows how this affects a gaussian distribution for illustration:

![image](/posts/hallucination/softmax.png)

Lower temperatures lead to sharper distributions: the highest probability distributions become more likely to be sampled. Conversely, higher temperatures mean the distributions are more uniform. When GPT-4 responds using a temperature of 1.0, we can access the less likely, extremes, of the distribution.

Note for the limiting case of $T=0$ we get an $argmax$ distribution; in other words, we take the most probable next word.

There's a relation here to google search that we're stating: The role that LLMs have started to fill is analogous to searching for something online. The result is not always perfect but we anticipate errors. Google searches, by their design cannot hallucinate articles or websites. If you ask for something that isn't on the internet, it won't return you anything. 

Note this was done using the instruction-tuned GPT-4, so the answers may differ from the regular ChatGPT interface.

## Conclusion

The issue with hallucinations is a feature of the transformer architecture. They're issues we will be able to fix, but not with bigger models or better training alone. They are, for now, an inherent issue with the way large language models attain and store information. Some day, it may be possible to inplant directly into the weights of a model specific facts, information or experiences, similar to how humans accumulate memories. The relationship between hallucinations and the concept of truth is nuanced and complex. 

Let's rephrase our discussion in another way. GPT-4 is undoubtedly an incredbily capable model. Today, complex frameworks consider it as a buildingblock for more complex orchestrated systems such as frameworks like AutoGen and GPTEngineer. These frameworks coordinate several agents which do simpler tasks themselves. So it's perhaps feasible that we could improve the performance of LLMs by stacking several LLMs on top of one another.

We discussed that inference by probability is inherently flawed. However, what if we instead made a host of models, whose task is solely to predict the next word. What if we made such systems to wrap around a language model? That is, if we ignore computational cost, could we do something like: "your task is to predict the next word in the sequence, explaining your reasoning". Here, we iteratively break down the question, and try to make the model build up some intuition and problem solving. The result of doing this on the task we discussed earlier (with the same parameters) removed our previous issue of the model repeating itself: 

```bash
GPT-4: Zibber 
GPT-4: flibber 
GPT-4: wobber 
GPT-4: gobber 
GPT-4: kibber
...
```

The language models we use today are perhaps increasingly considered better than an analogous google search, where we understand the result is not necessarily the truth but have been swayed to it's reliability with time. It's astonishing, the level of computer cognition we have access to. The lack of understanding and control we have over the cognitive behaviour of large language models reflects a general unease within the machine learning community; models are easy to train but hard to interpret, making models difficult to deploy. Efforts for interpretability in machine learning is an article for itself, but here we've touched upon how the scale of the complexity of the issue of hallucinations is complex to solve. Hallucinations serve as the inflection point for AGI, the point where statistical pattern matching becomes exact, robust semantic reasoning.