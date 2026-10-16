# Demo and viva notes

## 35-second project pitch

SentinelRL tests whether a DQN can defend a small synthetic enterprise network when attack propagation is stochastic, alerts are noisy and the defender has only six response actions. The agent can inspect, patch, isolate or deploy decoys. I compare three trained checkpoints against random, patch-priority, alert-responder and threshold policies, then test the same agents under stronger attack intensity. This is a simulation study: the current DQN improves over random and patch-only policies but does not beat the best defensive heuristics.

## One state-action-reward example

Suppose the database has criticality 1.0, an elevated belief score and a fresh alert, but has not been inspected. The action `inspect host 3` costs one of six response tokens and contributes -0.16 for the token. If the hidden host is compromised, inspection reveals it and adds +1.50; if clean, it subtracts 0.18. A later `isolate host 3` earns +1.10 + 0.55(1.0) if the compromise is real, but loses 0.65 + 0.85(1.0) if clean. The action has stochastic downstream value because isolation prevents that host from spreading further while consuming scarce budget.

## Likely questions

**Why RL?** Choices consume a limited budget and affect future attack spread, alerts and the value of later actions. The task is sequential and stochastic. The baseline results also show that RL is not automatically better than heuristics.

**Why is this a POMDP?** True compromise flags are hidden from the agent. It observes noisy alarms, action outcomes and approximate risk beliefs. The underlying simulator dynamics are Markovian, but the agent receives partial observations.

**Is the belief state exact?** No. It uses a factorized approximate update based on sensitivity and false-positive likelihood. This is explicitly a limitation.

**Why DQN?** Observations are numeric vectors and actions are discrete (25 action IDs with masking). DQN is straightforward to inspect and run on CPU. PPO or recurrent belief-history models would be future comparisons, not implemented claims.

**What makes the reward meaningful?** It gives credit for detecting and isolating real compromises and a decoy diversion; it penalizes compromises, persistent uncontained critical assets, response expenditure, clean-host isolation and false alarms. The coefficients are synthetic and must not be interpreted as calibrated business costs.

**Do results prove DQN works better?** No. It improves over random and patch-priority policies but loses to the alert-responder and threshold baselines on overall return and containment in these tests. This is a key result, not something to hide.

**What does the sensitivity plot show?** Critical compromises as attack probability increases from 0.18 to 0.28. DQN is not best across these scenarios. The test has only 20 episodes per condition.

**Can this protect a real enterprise?** No. The environment is a toy simulator without real traffic, malware, real alert correlation or operating-system actions. It is coursework experimentation only.

**What is the originality claim?** The contribution is the independently implemented resource-constrained, noisy-telemetry POMDP and evaluation, not a new RL algorithm or a first-ever cyber-defence method.

**How do you reproduce results?** Install requirements, run tests, then `python run_all.py --episodes 280 --eval-episodes 30`. Checkpoints, episode-level CSVs, summary tables and plots are written under `models/` and `results/`.
