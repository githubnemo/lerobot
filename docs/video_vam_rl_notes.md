# RL in Video-Action Models: what FLUX-mimic most likely does

Status: analysis/reconstruction, 2026-08-21. Source of truth is the FLUX-mimic
announcement (mimicrobotics.com/blog/introducing-flux-mimic, 23 Jul 2026),
section "Closing the Reinforcement Learning Post-Training Loop". mimic has NOT
published the algorithm; everything below the "verbatim clues" section is our
inference and should be labeled as such if cited.

## Verbatim clues from the blog

- "sample-efficient **off-policy RL** built on **pre-trained representations**"
- "we extend the policy architecture and **train a critic** to provide a
  **value function that scores states within rollouts**"
- the value function is used "to improve the policy **both during rollout,
  such as through best-of-n selection, and through ongoing policy refinement**"
- data source: "deployment provides a growing stream of data at no additional
  cost"; teleoperated demonstrations establish core behaviors first.

Note: RL appears only in FLUX-mimic (Jul 2026). The original mimic-video paper
(Dec 2025) — the recipe we replicate — is pure imitation.

## Our reconstruction

The clue combination (off-policy + critic + best-of-N + refinement, on a huge
flow-matching chunk decoder) maps onto **value-guided steering plus
advantage-filtered refinement**, not gradient-through-the-policy actor-critic:

1. **Critic training (off-policy).** A value function is trained on the
   deployment buffer (successes and failures) with an offline-RL-safe
   objective — IQL-style expectile regression or Cal-QL-style calibrated
   conservatism. Rewards are plausibly sparse task success (+ maybe
   time-to-completion shaping, since they emphasize speed-up beyond
   teleoperator limits).
2. **Test-time best-of-N (V-GPS pattern).** Sample N action chunks from the
   frozen generative policy, score the resulting states/chunks with the
   critic, execute the argmax. Improves behavior with zero weight updates.
   This is essentially V-GPS (Nakamoto et al. 2024, "Steering Your
   Generalists: Improving Robotic Foundation Models via Value Guidance"),
   which trains its value function with Cal-QL and reranks a generalist
   policy's action samples.
3. **Ongoing refinement.** Periodic fine-tuning of the action decoder with the
   SAME flow-matching loss used in pretraining, but on value-filtered or
   advantage-weighted deployment rollouts (AWR / filtered BC on action
   chunks). Off-policy, stable, cannot catastrophically break a working
   policy — the property you want on a factory line.

Why not direct Q-gradient through the policy (DDPG/SAC-style): the policy is a
large flow-matching chunk decoder; backpropagating a Q-gradient through the
sampling chain is expensive and unstable at this scale, and nothing in the
blog suggests it.

## The algorithms, properly explained

### IQL — Implicit Q-Learning (Kostrikov, Nair, Levine, 2021)

NOT "implicit quantile learning" (that is IQN, implicit quantile networks,
from distributional RL — unrelated).

Problem it solves: in offline/off-policy RL, the TD target
`r + gamma * max_a' Q(s', a')` queries Q at actions the dataset never
contains, and Q is garbage there (overestimation on OOD actions).

IQL's trick: never query Q at unseen actions. Instead learn a state-value
V(s) by **expectile regression** against Q(s,a) over the actions actually in
the dataset:

- `L_V = E[ |tau - 1(u<0)| * u^2 ]` with `u = Q(s,a) - V(s)`, tau ~ 0.7-0.9.
- With tau -> 1, V(s) approaches the max of Q over _dataset_ actions — an
  implicit max without ever evaluating OOD actions.
- Q is then trained with the safe target `r + gamma * V(s')`.
- Policy extraction (when needed) via advantage-weighted regression:
  `w = exp(beta * (Q - V))`, i.e. weighted imitation of good dataset actions.

For a best-of-N system you may only need V and/or Q — the "policy extraction"
step is replaced by reranking the generative policy's own samples.

### Cal-QL — Calibrated Q-Learning (Nakamoto et al., 2023)

Problem it solves: CQL-style conservative critics are great offline but, when
you continue training **online** (exactly mimic's deployment loop), the
over-conservative Q-values cause an initial performance _crash_ ("unlearning
dip") before recovering.

Cal-QL's trick: keep CQL's conservatism (push down Q on OOD actions) but
**calibrate** it — never let the learned Q sink below the value of a known
reference policy (in practice, Monte-Carlo returns of the behavior policy from
the buffer):

- CQL term: minimize Q on policy-sampled actions, maximize on dataset actions.
- Calibration: apply the push-down only where Q exceeds the reference value,
  i.e. penalize `max(Q(s,a_policy), V_ref(s))` instead of raw Q.
- Result: conservative enough to be safe offline, calibrated enough that
  online fine-tuning improves monotonically from step one.

That offline-to-online property is precisely the deployment-loop shape mimic
describes, which is why Cal-QL (or something in its family) is our best guess
for the critic objective. V-GPS uses Cal-QL for the same reason.

### What base does the critic sit on?

Unknown. The blog only says "built on pre-trained representations". Most
plausible reading: the critic is a **small head (MLP or shallow transformer)
on top of the FLUX-3 backbone's latent features** — the same features the
action decoder consumes — likely frozen or shared, so critic training is cheap
and sample-efficient (the representation already encodes physics/scene state;
the critic only learns "how good is this state for this task"). Alternatives
(separate vision encoder, proprio-only critic) are possible but would waste
the backbone they emphasize. Whether the critic scores states V(s), chunks
Q(s, a_chunk), or resulting predicted states is not disclosed; "scores states
within rollouts" leans V(s) evaluated along the rollout.

## Relevance to our project

Not for this week (we are pre-deployment). But the pattern is compute-friendly
for us later: a frozen VAM/SmolVLA policy + a tiny critic head on cached
backbone features + best-of-N chunk selection at rollout is implementable on
a 4090 and is the natural "day 2" improvement loop once real rollouts exist —
it needs only success labels per episode, which we get for free by running
the robot.

## References

- FLUX-mimic: https://www.mimicrobotics.com/blog/introducing-flux-mimic
- V-GPS: Nakamoto et al., "Steering Your Generalists", arXiv:2410.13816
- IQL: Kostrikov et al., arXiv:2110.06169
- Cal-QL: Nakamoto et al., arXiv:2303.05479
- Q-chunking / RLPD lineage for off-policy RL with demos: Ball et al.,
  arXiv:2302.02948
