# Results and interpretation

## Execution record

- Test suite: 6 tests passed.
- Training: 280 episodes each for seeds 17, 29 and 43; saved checkpoints in `models/`.
- Standard evaluation: 30 held-out episodes per baseline; 90 DQN model/episode combinations (three trained checkpoints times 30 evaluation seeds).
- Sensitivity: 20 held-out episodes per attack rate, with attack probabilities 0.18, 0.23, 0.28; three DQN checkpoints plus Alert Responder and Threshold Defense.
- Hardware/runtime: CPU only in the current execution environment; no GPU used.

## Standard evaluation snapshot

Metrics below are mean (standard deviation) over evaluation episodes as stored in `results/summary.csv`. For DQN, checkpoint outcomes are averaged within each of 30 held-out scenario seeds before computing the reported mean and standard deviation. Baseline summaries use one run per scenario seed.

| Policy | Mean return | Critical compromises | Uncontained at end | Mean detection delay | False isolations |
|---|---:|---:|---:|---:|---:|
| DQN | -7.335 (10.452) | 0.644 (0.748) | 1.244 (1.086) | 0.961 (1.415) | 0.211 (0.309) |
| Alert Responder | -2.573 (6.371) | 0.233 (0.504) | 0.433 (0.858) | 1.469 (1.033) | 0.000 (0.000) |
| Threshold Defense | -3.990 (9.855) | 0.400 (0.855) | 0.500 (1.167) | 1.593 (1.740) | 0.000 (0.000) |
| Patch Priority | -13.870 (19.235) | 0.933 (1.230) | 2.100 (2.537) | 1.153 (0.530) | 0.000 (0.000) |
| Random Legal | -15.583 (12.360) | 0.667 (0.922) | 1.933 (1.484) | 3.098 (8.690) | 1.200 (0.887) |

DQN improves on the random and patch-priority baselines in mean return, average critical compromises and mean uncontained compromises. It does **not** outperform Alert Responder or Threshold Defense on overall return or containment in this run. DQN's mean detection delay is lower than either of those baselines, but its episode-to-episode variability is high. This is an honest limitation and should be discussed rather than hidden.

All three seeds' mean training return improved between their first and last 20 episodes: seed 17 -16.252 to -6.886; seed 29 -11.062 to -5.114; seed 43 -15.080 to -8.246. These are training summaries, not evidence that the learned policy surpasses the strongest baseline.

## Attack-rate sensitivity

At attack probability 0.23, mean critical compromises were 0.483 for DQN, 0.050 for Alert Responder and 0.250 for Threshold Defense. At probability 0.28, they were 0.567, 0.400 and 0.500 respectively. DQN is not robustly best in the shift experiment. Sensitivity results are descriptive and use only 20 scenarios per condition.

## Interpretation guardrails

- Do not report results from unexecuted runs or alter the CSV values manually.
- Do not claim statistical significance: no hypothesis test has been run.
- Do not say the agent is suitable for production cybersecurity.
- Do not describe reward improvement as equivalent to a reduction in real incidents.
- If rerunning training, update the report figures and text from the new `results/*.csv` files.
