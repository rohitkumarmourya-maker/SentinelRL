# SentinelRL Champion

**Rohit Mourya (B23ES1029) · Jyotiraditya Singh Rathore (B23ES1017)**  
[Team's original GitHub repository](https://github.com/rohitkumarmourya-maker/SentinelRL)

A runnable reinforcement-learning project for budgeted network defense under noisy alerts. The champion uses a causal infection belief and learned Double DQN action values. Its deployment uniformly averages 10 snapshots: five independently trained seeds, each saved at 10k/20k episodes. It has no teacher or strategic action override. The environment/reward/budget/horizon are unchanged from the corrected project.

**Result:** higher episodic return than every original model/rule and every retrained RL comparator tested here, including each individual baseline checkpoint, using familywise-adjusted paired intervals. It does **not** establish superiority over the separately tuned stronger belief-threshold control. That control's mean return is slightly better and the adjusted difference is inconclusive. No claim of winning every metric or every possible policy is made.

## Install and run

Python 3.10–3.12, CPU; the recorded run used Python 3.12.5/PyTorch 2.8.0. From the extracted `SentinelRL/` folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py --steps 8
python -m pytest -q
python scripts/verify_champion.py --replay
```

Windows activation: `.venv\Scripts\activate`. No credentials, external dataset, network service, Gym or GPU is needed. The demo loads the actual bundled 10-member policy with verified source snapshots. All 57 tests passed before the final run; the verifier checks the sealed runtime, selected model files, immutable chunks and an exact recorded-episode replay. A new virtual-environment install was not repeated; dependency versions are recorded in `requirements-tested.txt`.

## Frozen benchmark

10,000 fresh paired scenarios,24 reported policies,520,000 actual policy/scenario episodes. The protocol, all weights and runtime were hash-frozen before parent-review authorization. There was no test-driven tuning, seed selection, member reweighting or outcome-based rerun. Primary return inference uses 50,000 crossed-bootstrap replicates and Bonferroni adjustment over 55 contrasts. The fixed 10-member ensemble is **one deployment policy**; its intervals are conditional on those trained members. Individual training-seed variability is reported separately. Original/checkpoint families retain their actual unequal seed counts.

| Common corrected-task policy | Mean return ↑ |
|---|---:|
| New validation-tuned threshold diagnostic | **−1.5567** |
| Champion temporal learned-Q ensemble | **−1.6373** |
| Original threshold rule, lawful adapter | −2.7499 |
| Original graph checkpoints, lawful transfer | −4.9601 |
| Matched pure graph temporal ensemble | −2.2131 |
| Graph+prior/shield, TD-only retrain | −4.8905 |
| Original tabular algorithm, lawful retrain | −4.1157 |
| Previous corrected DDQN | −7.4671 |

Champion minus matched graph ensemble: **+0.5759 [0.4594,0.6949]**. Champion minus original threshold: **+1.1126 [0.8834,1.3575]**. Champion minus tuned diagnostic: **−0.0805 [−0.2231,0.0632]**. Brackets are simultaneous conservative 95% familywise intervals. 54 of 55 contrasts have positive adjusted lower bounds; the tuned diagnostic is the sole exception. The complete table includes all original rules, original graph/no-stealth transfers and ensembles, original tabular transfer/retrain, previous corrected agents, matched graph retraining and individual fixed checkpoints.

The original GitHub return −.360 was faithfully reproduced separately, but belongs to a different leaky observation/reward design and is not directly compared with these scores. See `historical/original_github_reproduction/`. Original transferred models receive a lawful 58D adapter; unavailable hidden channels are removed and transfers are explicitly labeled. A proper same-task graph retrain prevents treating distribution shift alone as a victory.

Main result files are `results/final_benchmark/summary_intervals.csv`, `paired_comparisons.csv`, `per_seed_metrics.csv`, `episodes.csv` and 520 immutable chunk files. Every raw episode includes all 48 actions and rewards. The report is `report/main.pdf` with editable sources/figures. Critical compromises, false isolation, confirmation coverage and delay are reported honestly; return gains do not imply universal secondary-metric dominance.

## Reproduce

```sh
# Fresh champion training: exact frozen class/config, random initialization per seed
python train.py --execute-model-run --name reproduction_champion

# Quick reevaluation of bundled policies on 20 development scenarios
python evaluate.py --execute-model-run --workers 2 --output runs/quick-check

# Reproduce the entire unchanged recorded final benchmark
python evaluate.py --execute-model-run --full-frozen-test --workers 8 --output runs/full-check
python summarize_champion_benchmark.py --run runs/full-check --bootstrap 50000 --include-fixed-checkpoints
```

Use fresh output names; existing experiments are protected. Training saves genuine 10k/20k checkpoints for all five prespecified seeds. It uses the selected immutable runtime and discards the template's trained weights before initialization. Configuration selection and pruning were confined to development; all trial logs/source snapshots remain in `candidates/`, `baseline_candidates/` and `validation/`. `protocol/frozen_comparison.json` is authoritative; `config.json` is informational.

Pure graph, TD-only prior/shield graph, original tabular and exact historical tabular reproduction commands are recorded in their source runners/manifests. `docs/COMPARATORS.md` specifies adapters, budgets and optimizer differences. Recovery of two missing graph 10k snapshots used separate deterministic prefix replays matching every recorded numeric field; duplicate computation is explicitly recorded, never relabeled as new independent training.

## Source, history and portability

The public GitHub URL identifies the original team repository; these champion changes were **not pushed**. Original five commits are preserved, and one genuine current delivery commit is provided through a recoverable Git bundle after final review; no history/date/authorship was fabricated. Read `docs/HISTORY_AND_DATA.md` for restoration and data reconstruction.

The ZIP includes the complete consolidated raw CSV. Git history tracks its 16.7 MB gzip and all immutable chunks rather than a 344 MB single file. If restoring Git history, run `python scripts/reconstruct_results.py` before analysis. Source/runtime/model hashes and exact reconstruction checks preserve the evidence. Development data, historical results and the earlier corrected delivery remain clearly separated.

This is a six-host stationary teaching simulator with perfect explicit inspections and simplified costs. Its exact belief filter uses the known simulator model and public observations/actions only. The selected shared-host network has representational limits; better return here is not a claim of production cyber-defense safety, optimal POMDP control or superiority to every hand-designed policy.
