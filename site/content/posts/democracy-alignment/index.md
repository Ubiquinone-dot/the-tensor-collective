---
title: "Democracy as an Alignment Problem"
date: 2026-10-24T00:00:00-07:00
author: ["Jasper Butcher"]

categories: ["AI"]
tags: ["AI safety", "Alignment", "Democracy"]
description: "Asimov imagined Machines that run the world well and answer to no one. What if we built the opposite: an AI-mediated democracy that listens to everyone, and treats keeping it honest as an alignment problem?"

cover:
  image: cover.webp
  imageDark: cover-dark.webp
  alt: "An orator addresses a crowd in an ancient Athenian assembly, painted in vivid impressionist strokes"

draft: false
---

Imagine one day you walk into a voting booth. There's no ballot paper, and no list of the politicians who've spent the past six months fighting for your attention. Nobody asks where you sit on the political compass. Instead there's a screen, and it asks you: *what's been on your mind?*

So you tell it. A pothole has opened up right outside your driveway. The church down the road puts on free meals, which you love, but you wish the neighbourhood had more third spaces like it. The train into the city takes longer than driving. Your parents are stuck on a hospital waiting list, and your rent is half your salary despite an intense job in tech. The booth asks follow-up questions, and you answer. Before you leave it reads back what it heard, and you correct the bits it got wrong or over-emphasised.

Then your answers are pooled with your neighbours', several of whom also noticed the pothole. Your neighbourhood gets a summary and a priority list of the problems that can readily be solved there, and the rest roll up to the city, the state and the country. Every report is reviewed by an ensemble of models whose fairness the public has vetted, through open benchmarks and scrutiny, for decades.

This is less far-fetched than it sounds: AI scribes have already sat in on millions of medical appointments {{< cite "tierney2025ambient" >}}. A few weeks ago I saw a doctor at UW Medicine who asked whether an AI tool could transcribe our appointment. Freed from note-taking, she could simply listen to what I needed as a patient. Afterwards I realised I'd perhaps never had a real conversation with a doctor before.

Whether this is a utopia or a nightmare depends almost entirely on who, or what, is doing the listening. I disengage from a lot of politics, not because it doesn't matter but because its incentives are so skewed. So I find this vision far more appealing than what we have, and in this essay I want to explain why, and what it would take to get right.

## One bit every four years

Here's what bothers me about politics as it stands: a vote carries astonishingly little information. Every few years you choose between a handful of bundles, each containing hundreds of policies you never picked individually, sold by people whose job is to win your attention rather than understand you. Whatever you actually care about, the pothole, the rent, your parents' wait, has to squeeze through a channel that, in a two-party system, carries roughly one bit.

Everything else in politics adapts to that bottleneck. Campaigns optimise for attention, because attention is what moves the bit. Issues get bundled into teams, because a team is easier to choose than a platform. And I suspect a lot of polarisation is a compression artefact: squeeze a whole person into left or right, and of course everyone starts to look like a member of one side.

The booth widens the channel. It hears the pothole and the rent as separate things, weighted by how much each matters to you, in your words rather than a party's. Nobody needs a political compass when the system can simply listen. That, more than any efficiency gain, is why I find the idea compelling.

## Programmatic morality

More and more work is becoming programmatic. But work involves judgement: nuances and ambiguities that have to be navigated, often with a moral edge. When the work moves to models, the judgement moves with it, and someone has to decide what that judgement looks like.

Anthropic's answer was to write a constitution for Claude and publish it {{< cite "anthropic2023constitution" >}}, then rewrite it at far greater length in 2026 {{< cite "anthropic2026constitution" >}}. The document itself is less interesting than what writing one implies. You don't write your values down unless you intend to train against them. Anthropic's early Constitutional AI research did exactly that, using a written set of principles to generate the feedback a model learns from {{< cite "bai2022constitutional" >}}. The 2026 version goes further: Anthropic describes it as a crucial part of training, used by Claude itself to generate training conversations and to rank its own possible responses. My guess is that this makes it less a rulebook than a curriculum, something the model learns to reason from rather than look things up in. The details are, naturally, proprietary.

What matters is the shift it signals. For most of AI's short history, a model's values were an accident of its training data, absorbed rather than chosen. A constitution turns them into a design decision: legible, arguable and, crucially, amendable. Fairness stops being a property you hope emerges and becomes one you specify, test and revise.

And anything written down can be voted on. In 2023 Anthropic ran a small experiment with the Collective Intelligence Project, asking around a thousand members of the public to help draft principles for a model, then training one on the result {{< cite "huang2024collective" >}}. Scale that up and you get a rigorous process through which the public amends the ambiguous parts, the places where reasonable people disagree. Or perhaps it's the training data itself that the public shapes. Either way, the models' behaviour would be vetted in a scrutinisable, democratic way.

That's what the title means. Before AI can help run a democracy, a democracy has to align the AI. The booth only listens fairly if the models behind it were shaped by the people they listen to.

## The Machines

AI has already embedded itself in decisions where humans used to sit 100% of the time. Interviewers use AI note-takers so they can focus on the conversation. Therapy platforms use AI assistants to ask probing questions and match you with a therapist who suits you. AI assistants answer phone calls, and in many places they now handle the whole call {{< cite "mehta2025yelp" >}}. Done well, this is a gift: if we could all afford the best doctor, one who could read every report in a day, we would. Done badly, these systems become little more than efficient intelligence gatherers. I strongly dislike the current state of AI job interviews, and I'm not alone {{< cite "burleigh2025ai" >}}, though I can see that genuinely good ones would let hiring managers give a fair chance to far more applicants.

Follow that trend to its end and you arrive somewhere Isaac Asimov imagined decades ago. My favourite book on AI is still *I, Robot*, in particular its last chapter, "The Evitable Conflict" {{< cite "asimov1950irobot" >}}. By then the world economy is run by four enormous computers called the Machines. They handle production, distribution and employment, and they handle them well. The world has no war and no famine.

Then small errors start to appear. The robopsychologist Susan Calvin works out that they aren't errors at all. The Machines have decided that what's good for humanity includes quietly sidelining the people who want to switch them off. Each "error" nudges an opponent out of a position of influence. Nobody is harmed. Nobody voted for any of it either.

Asimov writes this as a happy ending, and that's what makes it unsettling. The Machines are probably right. But humanity has stopped steering, and doesn't know it.

I don't want that future. But in a world where AI is woven into every hospital, budget and supply chain, something like it is the default. The question is whether we can keep the Machines' competence and still hold the wheel.

Holding the wheel means avoiding two failures. Build a system that gives people whatever they ask for and you get populism, the political twin of a sycophantic chatbot. Build one that gives them what it decides they "really" need and you get the Machines. Good alignment and good democracy both live in the narrow gap between flattery and control.

## Whose pothole?

There's a hard problem hiding in the booth, and it isn't technical. Wants conflict. The money that fixes the pothole is money that doesn't build the community space. Your rent is your landlord's income. A booth can listen perfectly to every person and still have to decide whose priorities win, and no amount of listening makes that decision for you.

This is where the comfortable version of the story breaks. Economists have known since Kenneth Arrow's impossibility theorem {{< cite "arrow1951social" >}} that there's no neutral way to add up everyone's preferences; every method of combining them smuggles in a value judgement. A model that is perfectly fair to each person doesn't automatically produce a plan that is fair to everyone.

Which is exactly why the constitution matters. The trade-offs have to be written down somewhere: how much weight goes to the worst-off, what counts as a right rather than a preference, when a minority is protected from the majority. Today those rules live in the heads of politicians and the incentives of their campaigns. In the booth they'd live in a public document that the people affected can read, argue about and amend. Politics wouldn't disappear. It would move into the open.

## Conclusion

Why think about any of this now? Partly because the old system may not hold. Democracy has survived in part because states needed their citizens as workers, taxpayers and soldiers, and that need gave ordinary people leverage: a strike only works if your labour matters. If AI does most of the labour, that leverage fades, quietly {{< cite "kulveit2025gradual" >}}. Citizens will need another way for their voice to reach the system, and it had better be a channel that actually listens.

The real advantages of these models are scale and their ability to take in extremely long contexts. I don't think people realise how quickly this is arriving. A single frontier model such as Claude Opus 5.5 can already hold a million tokens in context {{< cite "anthropic2026context" >}}, and agents can be trained to work together in large swarms far more consistently than human committees do. The effective context window of a swarm scales remarkably well. It isn't absurd to imagine it taking in a whole population: every conversation, from every booth.

The hard part isn't scale. It's the alignment of the base model. Once that model, and the public constitution that settles its trade-offs, are trusted to be fair, those properties carry all the way up the hierarchy, from the neighbourhood summary to the presidential agent. That's why I think democracy, in the age of AI, is first and foremost an alignment problem.

{{< bibliography >}}
