# Actual champion evidence

`final_benchmark/` contains the frozen, authorized 10,000-scenario comparison:24policies,520,000 actual episodes, 520 immutable chunks and 55 familywise-adjusted primary-return contrasts. No algorithm or runtime change followed these outcomes. 54 contrasts have positive adjusted lower bounds; the new tuned belief-threshold diagnostic is the sole inconclusive exception and has a slightly better point estimate.

`episodes.csv` retains every actual action/reward sequence. `episodes.csv.gz` is its lossless Git-compatible copy; immutable chunks also reconstruct it exactly. `summary_intervals.csv`, `paired_comparisons.csv` and `per_seed_metrics.csv` expose all outcomes, including fixed baseline checkpoints and secondary metrics. The fixed ensemble's confidence intervals are conditional on its members; five individual 20k models provide separate seed-variability evidence.

The five `training_champion_seed*.csv` files contain all 20,000 training episodes, actual transition/update counts and episode-aggregated loss/gradient norms. Complete development selection, pruning, baseline training and deterministic-prefix recovery evidence is preserved in the corresponding working directories. Earlier corrected results and the separately reproduced original GitHub benchmark are archived, not mixed into these scores.
