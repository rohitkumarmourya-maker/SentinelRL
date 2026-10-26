# Original SentinelRL GitHub benchmark — reproduced

**Reproduction completed on 2026-10-26.** This directory is isolated from the corrected/trained delivery. It restores the user's original commit `dff57ab019fde89506ea2969d7c384ce532a2bbf` using the preserved complete Git bundle. The original tracked source, checkpoint weights, environment, reward, action logic and evaluation seeds were not changed.

All three supplied checkpoints (seeds 17, 29, 43) loaded strictly into real PyTorch 2.8.0 with no missing/unexpected keys and no compatibility patch. Each checkpoint records 13,440 steps/1,680 updates and 280 training episodes. This reproduction evaluates those supplied weights; it does not claim to independently reproduce their historical neural training. The original tabular model had no saved checkpoint, so the unmodified original ablation function trained it for 280 episodes per seed before evaluation.

The original main evaluation has 30 held-out scenario seeds `300000+97*i` for i=0..29. For learned policies, original statistics first average the three checkpoints within each scenario, then report the sample mean and sample standard deviation across the 30 scenario averages. Table entries below are **mean ± scenario SD**, not confidence intervals.

| Original policy | Return | Critical compromised | Uncontained | Detection delay | False isolation |
|---|---:|---:|---:|---:|---:|
| Alert responder | -2.573 ± 6.371 | 0.233 ± 0.504 | 0.433 ± 0.858 | 1.469 ± 1.033 | 0.000 ± 0.000 |
| Dueling DQN (No Stealth Prior) | -0.955 ± 4.872 | 0.167 ± 0.592 | 0.100 ± 0.403 | 0.085 ± 0.278 | 0.000 ± 0.000 |
| Graph-Dueling DQN (Full) | -0.360 ± 2.672 | 0.111 ± 0.364 | 0.044 ± 0.243 | 0.922 ± 3.677 | 0.000 ± 0.000 |
| Patch priority | -13.870 ± 19.235 | 0.933 ± 1.230 | 2.100 ± 2.537 | 1.153 ± 0.530 | 0.000 ± 0.000 |
| Random legal | -15.583 ± 12.360 | 0.667 ± 0.922 | 1.933 ± 1.484 | 3.098 ± 8.690 | 1.200 ± 0.887 |
| Tabular Q-Learning | -3.018 ± 5.161 | 0.133 ± 0.346 | 0.256 ± 0.635 | 39.504 ± 8.084 | 0.600 ± 0.355 |
| Threshold defense | -3.990 ± 9.855 | 0.400 ± 0.855 | 0.500 ± 1.167 | 1.593 ± 1.740 | 0.000 ± 0.000 |

**Reproduction matches the committed results:** all 136 rounded summary mean/SD cells match exactly at the committed three-decimal precision. The original main raw metrics differ only by at most 3.55×10⁻¹⁵ from CSV floating-point formatting. Full-model rows independently rerun inside the ablation are exactly identical to the main evaluation rows. There are 210 main raw rows and 270 ablation rows, or 390 unique policy/checkpoint/scenario rows after removing the duplicated full-model evaluation.

## Confidence and seed variation

`reproduction/original_method_summary.csv` adds 10,000-replicate scenario-bootstrap 95% intervals conditional on the three supplied checkpoints. The full-model mean-return interval is [−1.4034,0.4508]. It does not include uncertainty over a broader population of independently trained agents. `reproduction/per_seed_summary.csv` retains all individual seed mean/SD values. Full-model seed 17/29/43 mean returns are −.534393/−.534393/−.010360; critical compromises are .133333/.133333/.066667. Tabular seed means vary substantially (−.919493/−6.502007/−1.633953), which the original scenario-averaged summary can obscure.

## What this means

This establishes faithful execution of the original GitHub benchmark. It does **not** establish that the original policy is purely learned or a fair winner against the corrected agent. The original observation exposes actual entry/infection events indirectly through belief/history and counters; its learned Q output is clamped to ±.12 before a much larger handcrafted prior, and its safety shield changes action scores. It also uses different rewards and perfect decoy immunity. Original “detection” includes ground-truth-labeled alerts, unlike the corrected explicit-confirmation metric. Absolute returns and detection values therefore cannot be compared directly across the two environments.

The original NoStealth label disables the stealth detector/prior flag at inference on the same full checkpoint; it is not a separately retrained architecture. Its remaining graph/network, handcrafted prior and shield are preserved. All reported original controls are included; no optimization or tuning was performed.

## Reproduce and inspect

From a **fresh** restored clone of the same commit, copy these two reproduction helper scripts, then run:

```sh
python3 -u reproduce_original.py
python3 summarize_benchmark.py
```

The driver invokes the unmodified original `src.experiment.evaluate(episodes=30,model_seeds=(17,29,43))` and `evaluate_ablation(...)`, redirecting only output paths to `reproduction/`. It protects an existing completed reproduction from overwrite. CPU threads were set to 1 as an execution setting; no source/algorithm/reward/weight patch was necessary. Original tracked `results/` remain the reported evidence; `reproduction/` contains fresh measurements.

Actual runtime was 5.728 seconds total, including tabular training, inference, aggregation and additional bootstrap postprocessing in the driver. Python 3.12.5, NumPy 2.2.6, pandas 2.2.2, PyTorch 2.8.0, Matplotlib 3.9.2 on arm64 CPU. The original committed dependency snapshot listed newer versions; strict loading and numeric reproduction succeeded despite that difference.

Key files: `reproduction/benchmark.csv` (compact table), `evaluation_matrix.csv` (all 390 unique raw rows), `per_seed_summary.csv`, `scenario_averaged_metrics.csv`, `reported_vs_reproduced.csv`, and `metadata.json` (exact commands, versions, checkpoint hashes, configs, strict-load proof and timing). `reproduction/file-hashes.json` covers the fresh evidence and helper scripts. No GitHub push or changes to the corrected project occurred.
