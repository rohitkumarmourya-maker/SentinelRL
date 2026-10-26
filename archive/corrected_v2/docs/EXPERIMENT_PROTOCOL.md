# Frozen experiment protocol — executed unchanged

This plan was fixed before model execution and then run unchanged after explicit user authorization. Actual evidence is in the primary models/results directories; no held-out outcome drove tuning. `src/protocol.py` and the dataclasses are authoritative; `config.json` is an informational snapshot, not parsed by the runners.

## Training and held-out scenarios

- Train independent DDQN and tabular Q-learning agents with seeds 17, 29, 43 for 2,500 episodes each. Training scenario seed is `1000000*training_seed + episode_index`.
- The fixed final checkpoint is evaluated; no checkpoint selection or hyperparameter tuning on held-out scenarios is implemented. Any later tuning must add a separate validation split and preserve the untouched test scenarios.
- Evaluate on 100 scenarios `300000+97*i`, i=0..99, shared across policies and training-seed blocks. Exogenous attacker/sensor noise is pre-sampled independently of actions. Random-policy randomness is separate from the environment stream.
- Evaluate sensor sensitivity .35/.55/.75 on a separate 100-scenario set `400000+97*i`, reused across conditions. Training environment remains sensitivity .55 and false-positive rate .08. This is a sensor-shift test; attack-rate robustness is not implemented.
- Retrain a no-memory DDQN with the same seeds/budget/architecture, zeroing host EMA, last inspection result and inspection age. Current alerts and defense flags remain. The separately saved initial, untrained DDQN checks whether training improves performance. Both comparisons were executed for all three seeds; raw results and uncertainty are bundled.

DDQN uses the documented defaults. Tabular Q-learning uses the same full 25 actions and environment interactions, gamma .98, alpha .15, and epsilon 1→.05 over 80,000 steps. Its state key quantizes each host's alert/inspection/defense flags plus budget and four time bins. It is a deliberately coarse RL baseline; it omits continuous alert history and cannot establish that function approximation always wins.

## Controls

No-op and uniformly random legal actions diagnose task difficulty. A fixed observation-only alert responder inspects high alert/EMA scores and isolates hosts with positive explicit diagnoses. It is clearly labeled `alert_rule_diagnostic`, does not train the DQN and is never inserted into DQN decisions. The project's central solution is reinforcement learning; these non-RL controls only test whether learning adds value.

## Metrics and uncertainty

Primary: undiscounted episode return and number of critical hosts ever compromised. Secondary: total compromised hosts, active compromised hosts at the end, false/successful isolations, tokens spent, invalid actions, confirmation coverage and censored confirmation delay. Undetected infections contribute `horizon - first_compromise_time` to censored delay; detected ones use `first_confirmation_time - first_compromise_time`. Report confirmation coverage alongside delay so missed detections cannot disappear from the metric.

Keep all raw episode rows, each seed's metric means and action counts. Summary CSVs include SD of independent seed means. Deterministic controls are repeated only for pairing; their `independent_training_runs` is 0, not 3. The `seed_pairing_blocks` field records three matching comparison blocks. Random and untrained controls vary by policy initialization/randomness, not training.

Use 2,000 crossed-bootstrap replicates that independently resample training-seed rows and paired-scenario columns. Calculate pairwise intervals from **within-pair differences**, preserving shared environmental randomness. The 300 episode rows are not 300 independent trained agents. Three training seeds is a small sample; intervals are exploratory and cannot establish broad robustness. Report effect sizes and individual seed outcomes, including any losses to the alert rule or tabular baseline.

## Evidence and reproducibility

The completed run produced all artifacts described below. Runners write training episode/update logs, parameter hashes before/after training, actual step/update counts, elapsed wall time, versions, source/checkpoint hashes and final checkpoints. Final hashes must differ from initial ones after actual gradient updates. Model tests verify a gradient changes parameters, target semantics and safe checkpoint round-trip. The runners refuse to overwrite intended model/raw run artifacts; preserve failed runs and choose a fresh output directory.

Genuine training/comparison charts are bundled and were generated from the recorded CSVs. `scripts/plot_results.py` refuses to plot when evaluation data is missing. The final report includes actual measured outcomes and limitations. No legacy performance numbers were reused.
