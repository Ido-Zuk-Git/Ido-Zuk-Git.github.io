---
layout: post
title: "Why Simulation is Hard"
description: "Fixing the HGD problems"
date: 2026-07-01 09:00:00 +0000
---

<p class="post-subtitle">Fixing the HGD problems</p>


### Recap: what are we trying to solve
In [the last post]({% post_url 2026-06-16-the-hgd-algorithm %}) I talked about how to improve an autonomous agent (say, an AI that talks to customers as a salesperson) using the data we collect in production from the interactions between the AI and our customers. I presented the HGD (Human Gradient Descent) algorithm and its shortcomings: basically, it uses regression testing to "lock" the mechanisms the agent already solved, but it stops at local minima. From there I defined the desiderata for a substitute algorithm: it needs to be easy to automate, to look at the trajectory-level problem, to escape local minima, and to scale positively as we invest more compute.


### The obvious answer: simulate the user
I think that the obvious answer to all of these is **simulating the user**, and how it interacts with the agent. By that I mean having another system - an LLM? - play the user's side of the interaction with the agent. Recall from the appendix that to optimize the agent we needed the environment transition $P_e(s_{t+1}|s_t,a_t)$ - and in our case the environment is just the user. In the real world we actually have this $P_e$ - our real customers - but we can't run RL against them. So a simulator is really just a usable copy of $P_e$, one we're allowed to play with. 

Of course - this idea is not new. The most prominent example is autonomous driving, where Waymo, for example, uses [high-fidelity simulators](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/) to simulate how their AV interacts with the world. Although the fields are different, the idea is quite the same - whenever you have a long horizon (many turns/actions) of agent-human interaction, simulation is the only scalable way to develop and maintain these kinds of systems.

Presumably, this solves all the problems - given we have a good simulator of the user, we can run these simulations at scale, evaluate them, and improve the agent based on the results. The evals could be trajectory level and not only single turn, and the more simulations we run, the more we improve the agent, right? well... as anything with true value in this world, the devil is in the details. 
 

### Option 1: hand-designed test cases
The easiest option - and the one already offered by many vendors, though you could just as well build it in-house - is to invent the "test cases" yourself: an "angry user who doesn't like that the agent offered them a car to buy," or a "nice but irrational user who doesn't follow what the agent is saying." You write the personas, and a model role-plays them. There are three problems with this method.

1. **Bias.** The cases come from *your* imagination of what a user is, not from the real distribution $P_e$. Any eval you build on top of them is a biased estimator of how the agent actually performs.

2. **The Bitter Lesson.** Every case is a piece of hand-encoded human knowledge, which goes squarely against [Sutton's Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) - betting on human priors over computation and learning is usually a losing bet.

3. **You depend on the engine.** Say you defined the persona of an "angry user who doesn't like that the agent offered them a car to buy." How do you know the "engine" - the LLM meant to play it - actually follows that script, and in a human-like way?


### Option 2: off-policy evaluation - importance sampling
The next option, well known from recommendation systems, comes from the toolset of [Off-Policy Evaluation](https://mattlanders.net/off-policy-evaluation.html) (to be a bit more rigorous - everything in that post is OPE, but for simplicity I won't drag in all of the math here). The idea is to evaluate the *new* policy of our agent using data collected under the *old* policy - without ever deploying the new one. How do we do that?

Let's set up some notation, borrowing from the last post. Write $\pi_{\theta_{\text{old}}}$ for the **behavior policy** - the agent that was actually running in production and generated our logs - and $\pi_\theta$ for the **target policy**, the new agent we want to score. The value of a policy is just the reward we expect from its trajectories:

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi}\big[R(\tau)\big]
$$

We want $J(\pi_\theta)$, but every trajectory $\tau$ we logged was sampled from $\pi_{\theta_{\text{old}}}$. Importance sampling is the trick that reweights one expectation into the other:

$$
J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}}\!\left[\frac{P_{\pi_\theta}(\tau)}{P_{\pi_{\theta_{\text{old}}}}(\tau)}\,R(\tau)\right]
$$

Now recall that a trajectory's probability factorizes into the policy and the environment transition:

$$
P_\pi(\tau) = P(s_0)\prod_{t=0}^{T-1}\pi(a_t|s_t)\,P_e(s_{t+1}|s_t,a_t)
$$

Here is the nice part: in the ratio, the initial state $P(s_0)$ and every environment term $P_e$ are *the same* under both policies - so they cancel, and we are left with only the agent's own action probabilities:

$$
\frac{P_{\pi_\theta}(\tau)}{P_{\pi_{\theta_{\text{old}}}}(\tau)} = \prod_{t=0}^{T-1}\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

So our estimate of the new agent's value, straight from logged data, is a reward-weighted average over the $N$ logged conversations:

$$
\hat{J}(\pi_\theta) = \frac{1}{N}\sum_{i=1}^{N}\left(\prod_{t=0}^{T-1}\frac{\pi_\theta(a_t^i|s_t^i)}{\pi_{\theta_{\text{old}}}(a_t^i|s_t^i)}\right)R(\tau^i)
$$

This is quite nice. As long as the old policy $\pi_{\theta_{\text{old}}}$ covers every action the new policy might take - it sits in the denominator, so we can't have it be zero - we're good, and the estimator is even unbiased, which is great. So what are the problems?

The first is variance. It's easy to show on a toy example that the variance of this estimator scales with the *number of available actions* - and for an LLM agent, where an "action" is any sentence it could possibly say, that number is huge. But this one can be mitigated with a few mathematical tricks, so I won't call it a blocker.

The other two problems are more practical, and they come from the fact that these are LLM-based agents:

1. **How do we even define the action $a$?** Do we treat a specific utterance as a distinct action? The trouble is that many different utterances are really the *same* logical action - "please give me your date of birth" and "can you tell me what your date of birth is?" are the same move. How do we assign a probability to that whole class?

2. **How do we measure that probability?** Some closed-model APIs expose the log-probs, some don't - and rarely for every token. Without the behavior probabilities in the denominator, we can't even form the ratio.


### Option 3: learn the simulator
The last option, also conceptually simple, is to learn the user ourselves. We take our data of AI-user interactions and have a model "mimic" the user's side. Mathematically, it means learning an approximation $\hat{P}_\phi$ of the true environment $P_e$.

It seems like this approach sidesteps both the conceptual difficulty of Option 1 - the data is real, not from our imagination - and the practical difficulties of Option 2 - we own the simulator, so we control its probabilities. But it presents a fresh set of questions. How do we define the persona so that Richard Sutton would be satisfied? How do we handle the "long horizon" problem, where the more turns our agent takes against the simulator, the more the error compounds? Which data do we need for that, and how much of it? How do we deal with the covariate shift - the simulator is trained on data from today's agent, but the moment we improve the agent it starts visiting conversations the simulator has never seen? And how can we even tell whether the simulator works well?

All of these are crucial - and fascinating - questions for this method to actually work.
