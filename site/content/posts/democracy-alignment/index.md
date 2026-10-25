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
  source: "Painted from an image in Encyclopædia Britannica's *Democracy* gallery"
  sourceUrl: "https://www.britannica.com/topic/democracy/images-videos"

draft: false
---

Imagine one day you walk into a voting booth. There's no ballot paper, and no list of the politicians who've spent the past six months fighting for your attention. Nobody asks where you sit on the political compass. Instead there's a screen, and it asks you: *what's been on your mind?*

So you tell it. A pothole has opened up right outside your driveway. The church down the road puts on free meals, which you love, but you wish the neighbourhood had more third spaces like it. The train into the city takes longer than driving. Your parents are stuck on a hospital waiting list, and your rent is half your salary despite an intense job in tech. The booth asks follow-up questions, and you answer. Before you leave it reads back what it heard, and you correct the bits it got wrong or over-emphasised.

Then your answers are pooled with your neighbours', several of whom also noticed the pothole. Your neighbourhood gets a summary and a priority list of the problems that can readily be solved there, and the rest roll up to the city, the state and the country. Every report is reviewed by an ensemble of models whose fairness the public has vetted, through open benchmarks and scrutiny, for decades.

This is less far-fetched than it sounds. A few weeks ago I saw a doctor at UW Medicine who asked whether an AI tool could transcribe our appointment. Freed from note-taking, she could simply listen to what I needed as a patient. Afterwards I realised I'd perhaps never had a real conversation with a doctor before. AI transcribers like these have already sat in on millions of medical appointments {{< cite "tierney2025ambient" >}}. You also see AI models answering support calls {{< cite "mehta2025yelp" >}}, and BetterHelp using AI to help match patients with therapists {{< cite "lovett2023teladoc" >}}.

Whether this is really a utopia depends a lot on who's doing the listening. I disengage from a lot of politics, not because it doesn't matter but because its incentives are so skewed. So I find this vision far more appealing than what we have, and in this essay I want to explain why, and what it would take to get right.

## Your vote is currently one bit every four years

What bothers me about politics is that a vote carries astonishingly little information. Every few years you choose between a handful of representatives, each standing for hundreds of policies you never picked individually, sold by people whose job is to win your attention rather than understand you. Whatever you actually care about, the pothole, the rent, your parents' wait, has to squeeze through a channel that, in a two-party system, carries roughly one bit.

Everything else in politics adapts to that bottleneck. Campaigns optimise for attention, because attention is what moves the vote. Issues get bundled into teams, because a team is easier to choose than a platform or a set of nuanced decisions. The real crux, I suspect, is that a lot of polarisation arises from this: a system that incentivises squeezing a whole person into left or right. It wouldn't surprise me if fixing these divided systems were a first step towards unifying communities, and with them our sense of kinship.

Humans might be drawn to this polarisation: it's easy, and classifying others as right or wrong is a nice way to bond with friends. But in reality I think most people are well-intentioned. They form their political views from a complex mix of their life experience and history, all of which is very real and shouldn't be discounted on the basis of a political label.

The booth widens the information channel and lets individuals be represented as themselves. It hears the pothole and the rent as separate things, weighted by how much each matters to you, in your words rather than a party's. Nobody needs a political compass when the system can simply listen. It's not just an efficiency gain; it's an inherently more democratic system.

There's a second cost to votes being so sparse: someone has to do an ungodly amount of work filling in everything a single vote leaves out, and that someone is usually a politician. The true breadth of a politician's responsibilities, not to mention a president's or prime minister's, is almost superhuman. A booth that listens to everyone could take some of that weight off them.

So how would we get there? The rest of this essay looks at the makings of a machine we can trust, how such systems could scale to whole societies, and why that future probably won't arrive through the ballot box first.

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

Asimov doesn't write this as a happy ending, or as a sad one. He leaves the tension open: he never tells you whether it's a good idea, he just spells out a world in which it plays out. I like that, and it's what makes it unsettling. The Machines are probably right. But humanity has stopped steering, and doesn't know it.

I don't want that future. But in a world where AI is woven into every hospital, budget and supply chain, something like it is the default. The question is whether we can keep the Machines' competence and still hold the wheel.

Holding the wheel means avoiding two failures. Build a system that gives people whatever they ask for and you get populism, the political twin of a sycophantic chatbot. Build one that gives them what it decides they "really" need and you get the Machines. Good alignment and good democracy both live in the narrow gap between flattery and control.

## The Pothole Loopholes

The Machines aren't the only thing to worry about. A system like this, without human oversight, also has to survive human incentives. How would we ensure it isn't exploited by people? What if someone over-amplifies their problems, or secretly creates issues in their own lives to unlock the system's benefits? Every rule the booth uses to set priorities becomes something to game, and the people most motivated to game it are rarely the ones who most need help.

That's why I believe agent government is far more likely to arrive from the outside in, rather than from the inside out. Governments are likely to adapt well-established systems for allocating resources within individual departments first. First the healthcare system becomes standardised and more efficient, then transport, then housing. You might never go to the government about your parents' delayed appointments; instead, your doctor would already serve you the most efficient allocation of appointments.

That is, it's unlikely that systems will suddenly be run autonomously by agents. More likely, institutions will transition towards them one department at a time, and the question that matters most becomes the alignment of their models' behaviour in ambiguous circumstances.

## Conclusion

In the penultimate chapter of the book, "Evidence", a popular politician Stephen Byerley is running for office when his opponents accuse him of being a robot. To prove he's human, he punches a heckler at a rally, something the First Law would never allow a robot to do to a person, and the crowd loves him for it. Byerley wins, and by the book's final story he is World Co-ordinator, governing alongside the Machines. The robopsychologist comes to believe the heckler may have been a robot. Nobody ever finds out what he really was, and Asimov never tells us whether it matters. Maybe that's the right question for us too: not whether the thing in charge is a machine, but whether it's aligned with the people it serves.

Democracy has survived in part because states needed their citizens as workers, taxpayers and soldiers, and that need gave ordinary people leverage: a strike only works if your labour matters. If AI does most of the labour, that leverage fades, quietly {{< cite "kulveit2025gradual" >}}. We will need another way for people's voices to reach the system, and it had better be a channel that actually listens.

The real advantages of LLMs are scale and their ability to take in extremely long contexts. I don't think people realise how quickly this is arriving. Frontier models can already hold a million tokens in context. Agents can be trained to work together in large swarms far more consistently than human committees do. The effective context window of a swarm therefore scales unbelievably well. It isn't absurd to imagine an agent system orchestrating a whole population: every conversation, from every booth. The key point being this. Really, the hard part isn't scale. It's the alignment of the base model. Once that model, and the public constitution that settles its trade-offs, are trusted to be fair, those properties carry all the way up the hierarchy, from the neighbourhood summary to the presidential agent.

Progress is arriving faster than most people expect. In September, OpenAI announced that its models had produced a proof that the Navier–Stokes equations, given a suitable driving force, can develop a singularity in finite time, a result it says settles one of the seven Millennium Prize Problems {{< cite "openai2026navier" >}}. Mathematicians are still checking it, and OpenAI says it won't claim the prize. The surprising part is perhaps not that AI is more intelligent, but that well-established solutions to problems - and thereby institutions built around them - are about to be disrupted in the coming decades. Mathematics is the first sign of major disruption from this technology, and we could all reap the benefits in areas that directly affect us: infrastructure, logistics, healthcare, and eventually government itself. Imagine a state that allocates its resources as carefully as the best hospitals allocate beds, and can explain every decision it makes. That future is exciting, but only if we can trust whatever is doing the allocating.

{{< bibliography >}}
