# Actual revised experiment evidence

`STATUS.json` records completed training, inference, evaluation and all 35 passing tests. The user lifted the original hold with “run now”; no held-out result was used to change this experiment. Primary `models/` and `results/` are byte-identical copies of the completed `runs/full_20261026` experiment, with additional verification/facts files.

- `training_*.csv`: every training episode for 9 runs, each 2,500 episodes/120,000 transitions.
- `updates_*.csv`: every actual neural gradient update or tabular TD update, including finite loss/gradient norm or TD error.
- `manifest_*.json`: versions, exact source/checkpoint hashes, elapsed compute, actual counts and changed-parameter evidence.
- `evaluation_episodes.csv`:2,100 rawprimary rows (7 policies ×3 pairedseedblocks ×100 scenarios).
- `evaluation_per_seed.csv`: individual seed means; `evaluation_summary.csv`: mean/SD/95% crossed CI; `evaluation_paired_differences.csv`: uncertainty calculated on within-pair differences.
- `sensitivity_*`:6,300 rows and corresponding summaries for sensor sensitivity .35/.55/.75 on a separate scenario set.
- `action_type_means.csv`: action use, supporting the diagnostic-coverage limitation.
- `RUN_FACTS.json`: exact run counts/timings and verification outcomes.
- `replay_smoke.json`: saved seed 17 checkpoint reproduces scenario 300000's recorded metrics and 25 action counts exactly.
- `training_curve.png`, `evaluation_comparison.png`: plots from genuine CSV observations.

Three training seeds do not become 300 independent trained agents by evaluating 100 scenarios each. Deterministic controls repeat identical scenario outcomes solely to preserve pairing; their independent-training count is 0. Crossed bootstrap separately resamples training-seed rows and common-scenario columns. Intervals are exploratory with only 3 seeds. No test-set tuning, post-test retraining or best-seed selection was performed.

The main result shows learning relative to the initial network/random/tabular/no-op controls and an observation-history benefit; it does not establish an advantage over the alert rule. Explicit confirmation coverage is poor. See `docs/RESULTS_INTERPRETATION.md` for limitations and every comparison. Legacy outputs are archived under `../legacy/results/` and are not used in revised claims.
