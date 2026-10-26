# What the completed experiment establishes

This is the fixed, prespecified experiment: no reward/architecture/hyperparameter change, checkpoint selection, extra seed selection or test-driven tuning occurred after primary outcomes were inspected. All three final checkpoints are retained. No legacy performance number enters this analysis.

## Primary comparison

Double DQN mean return is −10.0418, 95% crossed-bootstrap interval [−13.5983,−6.8714]. Its three seed means are −10.5806,−11.0289,−8.5161; SD 1.3402. It has .5067 critical compromised hosts per episode [.3100,.7301], .2800 false isolations [.1900,.3867], and no invalid actions. Initial-network return is −43.1020, random −25.9554, tabular Q −31.8754, no-op −62.5561.

Paired DDQN return differences:

| Comparator | DDQN minus comparator | 95% crossed-bootstrap interval |
|---|---:|---:|
| Initial untrained network | +33.0601 | [26.5862, 39.9884] |
| Random legal actions | +15.9136 | [9.5730, 22.1637] |
| Tabular Q-learning | +21.8336 | [15.7692, 27.9455] |
| No action | +52.5143 | [49.0247, 55.5977] |
| Retrained no-memory DDQN | +4.0824 | [.9907, 7.8358] |
| Fixed alert rule | −.5593 | [−4.6983, 3.2485] |

This supports learning in this simulator and a benefit from the supplied compressed history features under the specified protocol. It does not establish superiority over the strong observation-only alert rule, whose mean return is −9.4826 and critical compromises .4600. The rule is a diagnostic comparator, never the DQN's teacher or deployment fallback.

## Diagnostic failure and action use

DQN explicit confirmation coverage is only .02644 (2.64%), with censored confirmation delay 45.4015 steps. The rule confirms .91133 (91.13%) with delay 5.0525. DQN mean actions/episode: inspect 1.3267, patch .9533, isolate 1.3433, decoy 1.0467 and wait 43.33. The rule inspects 4.64 and isolates 1.36.

Thus the agent does inspect, but usually does not successfully diagnose an actual infection before containment or episode end. It trades explicit confirmation for immediate isolation/protection from noisy evidence. The reward's small unconfirmed-active penalty ends on isolation, so high containment return does not guarantee a strong diagnostic workflow. No claim of faster detection or zero false isolation is justified. Future reward/representation studies would require a new validation split and a new untouched test protocol.

Censored delay depends on when infections occur; no-op can have a lower average than a containing policy because it creates many later infections. It should not be read without infection count and confirmation coverage, and does not alone establish a ranking of security outcomes.

## Sensor sensitivity

A separate 100-scenario set evaluates sensor sensitivity .35/.55/.75. DDQN return is −13.3039/−7.1574/−3.7978 and critical compromise count .7100/.3633/.1700. Better sensors improve observed control on this probe. The nominal .55 sweep result differs from the primary −10.0418 because it uses different environmental scenarios. The primary set remains the headline comparison; the more favorable secondary nominal condition is not substituted for it. All control outcomes and paired intervals are in the raw sensitivity CSVs.

## Evidence and limits

Every trained DDQN changes its parameter hash after 29,751 actual gradient updates. The three primary seeds use 360,000 transitions and 44.949 logged training seconds on an Apple M5 Max CPU, one PyTorch thread. Six neural runs plus three tabular runs use 1,080,000 transitions and 111.384 logged training seconds in total. These totals exclude setup/evaluation/plotting and are not general runtime promises.

All 35 unit tests pass. A saved-checkpoint replay exactly matches one raw held-out episode. Raw losses, gradient norms, TD errors, action counts, per-seed outcomes, bootstrap intervals and checkpoint/source hashes are provided. Those checks establish internal run consistency, not real-network effectiveness.

The experiment has three training seeds, fixed topology, a stationary nonadaptive attacker, perfect explicit inspections, immutable compromised state, a known entry distribution and simplified patch/decoy economics. A feed-forward network receives compressed history rather than an exact POMDP belief. Results do not establish deployment safety, calibration, broad robustness or superiority over expert operational procedures.
