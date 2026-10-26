# Exact revised design

SentinelRL is a teaching simulator. It does not connect to, scan, exploit or modify real systems. The network has six hosts and seven undirected edges: (0, 1), (1, 2), (2, 3), (1, 4), (4, 2), (3, 5), (2, 5). Host names and fixed vulnerability/criticality values are defined in `src/environment.py`.

## State, observation and actions

The reward-relevant augmented Markov state has nine dynamic quantities per host: compromise, patch, isolation, decoy, current alert, alert EMA, last inspection result, last inspection time and ever-confirmed flag; budget/time add two quantities (56 components). Physical compromise/deployment flags account for 24 bits. Additional event times are retained solely for diagnostic metrics. Fixed graph and host metadata are constants. Independent transition/sensor draws generate stochastic dynamics; pre-sampling them is an equivalent reproducible implementation.

The observation is a different 56-dimensional float 32 vector, bounded in [0, 1]. For each host in index order, its nine consecutive entries are:

| Offset | Feature | Meaning |
|---|---|---|
| 0 | current alert | Noisy binary sensor |
| 1 | alert EMA | .65 previous EMA + .35 current alert; starts at 0 before first sensor |
| 2 | last inspection | unknown=0, clean=.5, compromised=1 |
| 3 | inspection age | (current time − inspection time)/48; unknown=1 |
| 4 | vulnerability | Fixed known target susceptibility |
| 5 | criticality | Fixed known asset importance |
| 6, 7, 8 | patched, isolated, decoy | Defender's own persistent deployment flags |

The last two entries are remaining budget/6 and time/48. Neither actual entry identity, true compromised bits, true-positive counts nor unseen compromise events appear. Hidden infection does not reset inspection history or alter masks. A past clean diagnosis can become stale. This compressed history is not claimed to be a sufficient Bayesian belief; a feed-forward DQN only approximates a POMDP controller.

Actions are WAIT=0; INSPECT(i)=1+i; PATCH(i)=7+i; ISOLATE(i)=13+i; DECOY(i)=19+i for i=0..5. Non-wait valid actions spend one of six tokens. Masks depend only on budget, isolated flags and previous patch/decoy deployments. Reinspection is legal while budget remains. Waiting is always available. Invalid unavailable actions spend no budget and incur a penalty. Out-of-range/noninteger actions raise an error. Once the 48-step episode ends, another step requires reset.

## Transition and sensing

The initial foothold is hidden host 0 with probability .62, otherwise host 4. This distribution is known; its sampled outcome is not. The initial observation is produced by the same noisy sensor used later.

An action takes effect first. An inspection returns a perfect pre-spread snapshot; patch/decoy deployment changes future susceptibility; isolation removes all incoming/outgoing attack edges. No action cures an existing compromise. Then previously compromised nonisolated hosts simultaneously try to infect each still-clean nonisolated neighbor j, independently per source/target:

`p_j = .23 * vulnerability_j * (.20 if patched else 1) * (.25 if decoy else 1)`.

New infections can spread only next step. With k active neighbors, target infection probability is `1-(1-p_j)^k`. Both defenses can stack for multiplier .05; neither creates perfect immunity. Patching is a stronger first-use prevention than decoy at the same abstract token cost, so this simulator does not establish realistic relative deployment economics.

After spread, time advances and each nonisolated host emits an alert: P(alert|infected)=.55, P(alert|clean)=.08. Isolated hosts emit no alerts. Alerts and EMA are observable; true/false sensor labels are not exposed. Reset pre-samples the entire edge/time and sensor/time random fields so actions never change which random numbers are consumed. This permits paired future evaluations across action sequences and sensor conditions.

## Reward

All terms are summed after the action and spread:

`r = -.16*valid_nonwait -.45*invalid + .40*first_confirmation`
`    - sum_new_infections(2+4*criticality)`
`    - .16*sum_active_infections(criticality)`
`    - .04*sum_active_unconfirmed_infections(criticality)`
`    - sum_new_clean_isolations(.65+.85*criticality)`
`    - .03*sum_all_clean_isolated(criticality)`
`    - 1.4*number_active_critical_infections_at_terminal`.

“Critical” means criticality ≥.80. Active means compromised and not isolated. First confirmation is the first ever successful explicit inspection of that host; repeated inspections cannot farm reward. The undetected-active term charges an area-under-delay penalty until explicit confirmation or containment. No direct positive reward is given for patching, decoys or true isolation. Hidden ground truth may calculate simulator reward/metrics; the policy only receives the visible observation and legality mask.

## Learned controller

Two 128-unit ReLU hidden layers map 56 features to 25 unconstrained Q values. The learned Q output alone determines greedy action ranking among legal actions. Replay buffer 50,000, batch 64, gamma .98, Adam lr .0003, Huber loss, norm clip 10, warm-up 1,000 steps, update every four environment steps, target copy every 500 gradient updates. Epsilon decreases linearly 1→.05 over 80,000 environment steps. Double DQN online action selection and target evaluation both respect legal next actions. Terminal transitions have zero bootstrap because time is part of the finite-horizon observation. There are no demonstrations, fixed Q priors, teachers or strategic postprocessing shields.

Real forward/backward/checkpoint operations passed the four model tests and full training/evaluation protocol. All 35 tests pass by default. Six trained neural agents and three tabular agents, every raw update and 8,400 held-out rows are bundled; empirical limitations are documented separately.
