---
title: "Democracy as an Alignment Problem"
date: 2026-10-24T00:00:00-07:00
author: ["Jasper Butcher"]

categories: ["AI"]
tags: ["AI safety", "Alignment", "Democracy"]
description: "Asimov imagined Machines that would run the world but answer to no one. Today this future isn’t so far from reality, so what does it look like for humanity to maintain its voice?"

cover:
  image: cover.webp
  imageDark: cover-dark.webp
  alt: "An orator addresses a crowd in an ancient Athenian assembly, painted in vivid impressionist strokes"
  source: "Painted from an image in Encyclopædia Britannica's *Democracy* gallery"
  sourceUrl: "https://www.britannica.com/topic/democracy/images-videos"

substack: "https://jasperbutcher.substack.com/p/democracy-as-an-alignment-problem"

draft: false
---

Imagine one day you walk into a voting booth. There’s no ballot paper, and no list of politicians who’ve spent the past six months fighting for your attention. Nobody asks where you sit on the political compass. Instead there’s a screen, and it asks you: *what’s been on your mind?*

So you tell it. A pothole has opened up right outside your driveway. The church down the road puts on free meals, which you love, but you wish the neighbourhood had more third spaces like it. The train into the city takes longer than driving. Your parents are stuck on a hospital waiting list, and your rent is half your salary despite an intense job in tech. The booth asks follow-up questions, and you answer. Before you leave it reads back what it heard, and you correct the bits it got wrong or over-emphasised.

Then your answers are pooled with your neighbours’, several of whom also noticed the pothole. Your neighbourhood gets a summary and a priority list of the problems that can readily be solved there, and the rest roll up to the city, the state and the country. Every report is reviewed by an ensemble of models whose fairness the public has vetted, through open benchmarks and scrutiny, for decades.

This is less far-fetched than it sounds. A few weeks ago I saw a doctor at UW Medicine who asked whether an AI tool could transcribe our appointment. Freed from note-taking, she could simply listen to what I needed as a patient. Afterwards I realised I’d perhaps never had a real conversation with a doctor before. AI transcribers like these have already sat in on millions of medical appointments {{< cite "tierney2025ambient" >}}. You also see AI models answering support calls {{< cite "mehta2025yelp" >}}, and BetterHelp using AI to help match patients with therapists {{< cite "lovett2023teladoc" >}}.

Whether this is really a utopia depends a lot on who’s doing the listening. I disengage from a lot of politics, not because it doesn’t matter but because its incentives are so skewed. So I find this vision far more appealing than what we have, and in this essay I want to explain why, and what it would take to get right.

## The Machines

The book *I, Robot*, by Isaac Asimov {{< cite "asimov1950irobot" >}} walks the reader through a society that leans more on artificial intelligence with each chapter. In his world, robots are bound by the same three principles: (i) a robot may not injure a human being; (ii) a robot must obey the orders humans give it, except where they would conflict with (i); and (iii) a robot must protect its own existence, as long as that doesn’t conflict with rules (i) and (ii).

In the first chapter, named after the friendly robot “Robbie”, Asimov describes how a peaceful robot integrates itself as a friendly companion to the daughter of a family. By the last chapter of the book, “The Evitable Conflict”, the world economy is run by four enormous computers - *the Machines*. They handle production, distribution and employment, and they handle them well. Everything is run practically as efficiently as the laws of nature allow; the world has no war and no famine.

The Machines report calculations that inform political decisions to the World Co-ordinator. In one story, errors in a calculation produced by the Machines led to anti-machine activist Villafranca of growing influence being demoted in his organisation. In an investigation, robo-psychologist Susan Calvin discovers that these errors are actually intentional; the Machines have decided that what’s good for humanity includes quietly sidelining the people who want to rebel against them, nudging potential opponents like Villafranca out of positions of influence.

Perhaps surprisingly, Asimov writes this as neither a happy nor sad ending. As per the first law the Machines never really *harm* anyone. But no one (barring Susan Calvin and the World Co-ordinator) knows how the three widely accepted laws of robotics - and thereby governance - are actually playing out behind the scenes. The tension is that humanity lives in prosperity, yet has unknowingly stopped steering its course.

Details in fiction break down in reality, but undeniably many parts of this system are prosperous for humanity given everyone’s quality of life. A world where AI is woven into every hospital, budget and supply chain is certainly scary, but also unclogs many of the institutional inefficiencies that politics is meant to solve. The difference is that in Asimov’s world, humans aren’t listened to; the Machines have jumped ahead and left democracy behind. In his world, there’s no booth.

## Your vote is currently one bit every four years

What bothers me about politics is that a vote carries astonishingly little information. Every few years you choose between a handful of representatives, each standing for hundreds of policies you never picked individually, sold by people whose job is to win your attention rather than understand you. Whatever you actually care about, the pothole, the rent, your parents’ wait, has to squeeze through a channel that, in a two-party system, carries roughly one bit.

Everything else in politics adapts to that bottleneck. Campaigns optimise for attention, because attention is what moves the vote. Issues get bundled into teams, because a team is easier to choose than a platform or a set of nuanced decisions. The real crux, I suspect, is that a lot of polarisation arises from this: a system that incentivises squeezing a whole person into left or right.

Humans might be drawn to this polarisation: it’s easy, and classifying others as right or wrong is a nice way to bond with friends. In reality, as anyone who’s had a fruitful political discussion will know, most people are well-intentioned, regardless of political leaning. Everyone forms their political views as a complex concoction of their life experience and history, all of which shouldn’t be discounted on the basis of a political label. It wouldn’t surprise me if fixing a system that incentivises division marked a first step towards unifying communities, and with them our sense of kinship.

The booth widens the information channel and lets individuals be represented as themselves. It hears the pothole and the rent as separate things, weighted by how much each matters to you, in your words rather than a party’s. It’s not just an efficiency gain; it’s an inherently more democratic system.

There’s a second cost to votes being so sparse: someone has to do an ungodly amount of work filling in everything a single vote leaves out, and that someone is usually a politician. The true breadth of a politician’s responsibilities, not to mention a president’s or prime minister’s, is almost superhuman. A booth that listens to everyone, backed by a well-orchestrated and fair system, could take some of that weight off them. The question is, then, how to build such a system fairly.

## Fixing Pothole Loopholes

The Machines aren’t the only thing to worry about. In the case of agents interacting with the public, consider what happens if someone over-amplifies their problems, or *creates* issues in their own lives to access additional benefits. Every rule the booth uses to set priorities becomes something to game.

That’s why I believe an agentic government like the booth is far more likely to arrive from the outside in, rather than from the inside out. A swarm of agents running the state would have major limitations on appropriate alignment of behaviour. Instead, we are far more likely to build up autonomous institutions iteratively that each get incorporated into government. It starts with the report my doctor didn’t have to write, then the appointment scheduling, pharmacy and the rest are run on a flywheel within healthcare. The system becomes standardised and efficient, and can just do more depending on the customer’s needs. You don’t have to have a human to pull strings behind the desk to get you that earlier appointment that’s not on the calendar yet; the optimal customer experience is just provided.

Similarly to this, the transport system is automated and improved, then housing. You might never go to the government about your parents’ delayed appointments; instead, your hospital would already work as efficiently as possible, and tell the government directly about its resource constraints and demands. A hospital system like that works from records and test results rather than self-reported complaints, which makes it far harder to game, and preventing gaming of the well-oiled machine becomes the institutions’ main problem - and *that* is a well-posed alignment problem.

So, I don’t see it likely that systems will suddenly be run autonomously by agents. More likely, institutions will transition towards them one department at a time, and the question that matters most becomes the alignment of AI behaviour in ambiguous circumstances. For that, we need good systems of alignment.

## Programmatic and accountable judgement

When I talk to an AI over a support line, or get follow-up questions back from a form that is reviewed by AI, I have an almost unreasonable trust that a certain amount of attention will be paid to the effort I put in. Programmatically I know the mechanisms through which attention scores are computed between tokens in the context window I provide in my request. Perhaps the only thing I do not transparently know when initially interacting with an agent is which tools the company has hooked it up to, and how much it can reliably do before a human needs to get involved. In any case, I often give these systems far more detail than I’d ever type into a form I suspect nobody reads.

More and more of this type of work is becoming automated. But real work involves judgement: nuances and ambiguities that have to be navigated, often with an ambiguous - or sometimes moral - edge. When the work moves to models, judgement moves with it, and someone has to decide what that judgement looks like. I think it’s worth distinguishing safety from judgement here. Sure, we want agents that don’t go off-rails and harm people; that’s the first law. But we can get them to do something much larger than that; we want to instill reliability and judgement in as wide a scope as possible. That’s the second and third laws.

Anthropic’s answer was to write a constitution for Claude (published first in {{< cite "anthropic2023constitution" >}}, and rewritten at greater length in {{< cite "anthropic2026constitution" >}}). The document itself is less interesting than how it’s used: as a rubric that Claude’s values are aligned against, either directly or by generating derived training data {{< cite "bai2022constitutional" >}}. For most of AI’s short history, a model’s values were an accident of its training data, absorbed and emergent rather than chosen. A constitution turns them into a design decision that is open to public scrutiny.

As a result, a constitution like this can be voted on, and Anthropic has already tried a small version of exactly that, asking around a thousand members of the public to help draft principles for a model and then training one on the result {{< cite "huang2024collective" >}}. It can also be made arbitrarily detailed, provided the (currently proprietary) way it’s incorporated into training is robust. Perhaps the public shapes the training data itself, or the benchmarks used to vet the models in these embedded: a kind of public shaping of AI politicians, rather than a choice between a set of candidates.

Alignment emerges as a key problem with integration into society, with democratic fairness as its goal. Before AI can help run a democracy, a democracy has to align the AI (and likely do so continuously). The booth only listens fairly if the models behind it were shaped by the people they listen to. Build a system that gives people whatever they ask for and you get populism, the political twin of a sycophantic chatbot. Build one that gives them what it decides they really need and you get the Machines. Alignment and democracy both live in the narrow gap between flattery and control.

## Conclusion

Democracy has survived in part because states needed their citizens as workers, taxpayers and soldiers, and that need gave ordinary people leverage: a strike only works if your labour matters. If AI does most of the labour, that leverage fades, quietly {{< cite "kulveit2025gradual" >}}. We will need another way for people’s voices to reach the system, and it had better be a channel that actually listens.

AI has already embedded itself in decisions where humans used to sit. Done well, this is a gift. Done badly, these models become the front-page instruments of a dystopian society. I strongly dislike the current state of AI job interviews, and I’m not alone {{< cite "burleigh2025ai" >}}, though I can see that genuinely good ones would let hiring managers give a fair chance to far more applicants.

Scaling inference for agents isn’t the hard part. Frontier models can already hold a million tokens in context, and agents can be trained to work together in large swarms far more consistently than human committees do. It isn’t absurd to imagine an agent system orchestrating and representing a whole population: every conversation, from every booth. The hard part is the alignment of the base models. A public constitution that settles their trade-offs, and is trusted to be fair, can carry fairness properties all the way up the hierarchy and across systems, from the neighbourhood summary to the presidential agent, in an entirely new future for representation of the people’s needs.

In September, OpenAI announced that its models had produced a proof to the Navier–Stokes equations, one of the seven Millennium Prize Problems {{< cite "openai2026navier" >}}. The surprising part is perhaps not that AI is more intelligent, but that well-established solutions to problems - and thereby institutions built around them - are about to be disrupted in the coming decades. Mathematics is the first sign of major disruption from this technology, and we could all reap the benefits in areas that directly affect us: infrastructure, logistics, healthcare, and eventually government itself. Imagine a state that allocates its resources as carefully as the best hospitals allocate beds, and can explain every decision it makes. In *I, Robot*, nobody ever proves conclusively whether Stephen Byerley, the World Co-ordinator, is a human or a robot, but Asimov never tells us that it matters. Maybe that’s the right question for us too: not whether the thing doing the allocating is a machine, but whether it’s aligned with the people it serves.

-Jasper, 27 Oct 2026

{{< bibliography >}}
