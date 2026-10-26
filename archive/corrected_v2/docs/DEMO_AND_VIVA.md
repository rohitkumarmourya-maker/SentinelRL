# Demonstration and viva guide

Start with the actual result: “We trained an observation-only Double DQN on a small stochastic network-defense simulator. It learns better returns than its initial network, random actions and tabular Q, but it does not establish superiority over a fixed alert rule and rarely successfully confirms infections. All three final seeds and raw results are included.”

## Show the runnable work

```sh
python -m pytest -q
python demo.py --checkpoint models/ddqn_history_seed17.pt --execute-model-run
python scripts/verify_evidence.py --replay
```

The 35 tests include real gradients, terminal Double DQN targets, legal masks and serialization. The saved seed 17 demo is a fixed convention, not the best test seed. The replay check matches a specific recorded held-out scenario exactly. `python demo.py` alone is an environment API exercise; it must not be represented as learned-agent behavior.

For a new evaluation using the bundled weights, use a fresh directory:

```sh
python scripts/evaluate_pretrained.py --execute-model-run --output runs/fresh-evaluation
```

To rerun the full training protocol, use the fresh-directory command in the root README. The report and genuine learning/comparison plots are already included; display raw CSV and per-seed tables when discussing results.

## Explain the decision problem

A defender has six tokens over 48 steps. Attacks spread from a hidden foothold over seven edges. A .55-sensitive sensor emits .08 false alarms. Patching does not cure infection; isolation disrupts clean hosts. Actions influence future exposure, information and budget, so they require sequential rather than supervised per-step decisions.

A 56-feature observation contains six hosts × nine visible telemetry/history/metadata/deployment features plus normalized time/budget. The true compromised vector never reaches the policy. Clean inspections can become stale; hidden infections cannot reset visible histories or masks. This is compressed history, not an exact sufficient Bayesian belief.

There are 25 actions: wait plus inspect/patch/isolate/decoy on six hosts. The availability mask checks only public flags/budget, never latent infection or strategic preferences. The learned Q values alone rank legal actions. Real PyTorch gradients fit temporal-difference error from replay; online DDQN chooses the next action and its target network evaluates it, with zero terminal bootstrap.

## Show the scientific evidence and limits

Main return: DDQN −10.04 versus initial −43.10, random −25.96, tabular−31.88 and no-op −62.56. Its paired improvement over initial is +33.06[26.59, 39.99]; over no-memory +4.08[.99, 7.84]. The alert rule achieves −9.48: DDQN difference −.56[−4.70, 3.25], so no supported primary superiority.

DDQN successful explicit-confirmation coverage is 2.64%, versus 91.13% for the rule. It inspects 1.33 times/episode versus 4.64 for the rule, but mostly does not successfully diagnose infections. Direct noisy-evidence containment and prevention can yield return without a good diagnostic workflow; the small unconfirmed-active penalty stops upon isolation. This is a limitation to explain, not hide.

## Answer common questions

- **Why reinforcement learning?** Reward reflects future damage/service costs, not optimal-action labels. A full-action tabular RL baseline checks whether neural approximation adds value.
- **Does reward use hidden state?** The simulator may calculate reward from latent infections. Decisions receive only visible observations and legality masks; no privileged info or previous reward is a policy feature.
- **Can inspection farm reward?** The +.4 confirmation bonus occurs at most once per host. Each intervention still costs a token and −.16.
- **Are decoys perfect?** No: risk multiplier .25, patch .20, combined .05. None cures infection. Relative economics are simplified.
- **Can isolate-all trivialize the task?** It consumes all six tokens and incurs immediate/continuing clean-isolation costs. Learned action distributions and the strong rule comparator reveal the actual tradeoffs; no optimality claim is made.
- **How is evaluation paired?** Pre-sampled edge/time and sensor/time random fields are independent of actions; policies share scenarios, with separate policy randomness. Bootstrap resamples trained seed rows and common scenario columns; 300 rows are not 300 trained agents.
- **Why does nominal sensitivity return differ between tables?** The separate sensor probe uses a different scenario set. Primary outcomes remain primary; favorable secondary numbers are not substituted.
- **What did original results prove?** They are historical artifacts only; hidden-state leaks and handcrafted action dominance prevent validating the corrected system with them.
- **What remains uncertain?** Only three seeds, one fixed topology, stationary attacks, perfect inspections, compressed feed-forward history and simplified deployment costs limit real-world transfer. Further changes require independent validation and untouched test scenarios.
