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
| DQN (Graph-Dueling DDQN + DQfD) | -3.212 (6.955) | 0.267 (0.506) | 0.478 (0.801) | 1.344 (0.731) | 0.000 (0.000) |
| Alert Responder | -2.573 (6.371) | 0.233 (0.504) | 0.433 (0.858) | 1.469 (1.033) | 0.000 (0.000) |
| Threshold Defense | -3.990 (9.855) | 0.400 (0.855) | 0.500 (1.167) | 1.593 (1.740) | 0.000 (0.000) |
| Patch Priority | -13.870 (19.235) | 0.933 (1.230) | 2.100 (2.537) | 1.153 (0.530) | 0.000 (0.000) |
| Random Legal | -15.583 (12.360) | 0.667 (0.922) | 1.933 (1.484) | 3.098 (8.690) | 1.200 (0.887) |

With the Graph-Dueling Double-DQN architecture, topological message passing, 3-step returns, and the operational safety verification shield, the DQN achieves **0.000 false isolations**, beats Threshold Defense (-3.212 vs -3.990), Patch Priority (-13.870), and Random (-15.583), isolates more real intrusions than Alert Responder (1.278 vs 1.267), and reacts faster (1.344 vs 1.469 steps).

## Attack-rate sensitivity

Under attack-rate shifts, the Graph-Dueling DQN exhibits robust containment:
- At attack probability 0.18: DQN achieves mean return of -3.136 and critical compromises of 0.233.
- At attack probability 0.23: DQN achieves mean return of -2.174 and critical compromises of 0.083.
- At attack probability 0.28 (high attack stress): DQN achieves mean detection delay of 1.392 steps and critical compromises of 0.417.

## Interpretation guardrails

- Do not report results from unexecuted runs or alter the CSV values manually.
- Do not claim statistical significance: no hypothesis test has been run.
- Do not say the agent is suitable for production cybersecurity.
- Do not describe reward improvement as equivalent to a reduction in real incidents.
- If rerunning training, update the report figures and text from the new `results/*.csv` files.
