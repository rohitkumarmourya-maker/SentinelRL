#!/usr/bin/env python3
"""Postprocess raw reproduction evidence without rerunning a policy or model."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reproduction'
METRICS = ['episode_return','critical_compromises','active_compromised_at_end','mean_detection_delay','false_isolations']
matrix = pd.read_csv(OUT/'evaluation_matrix.csv')
per_seed = matrix.groupby(['policy','model_seed'])[METRICS].agg(['mean','std'])
per_seed.columns = ['_'.join(column) for column in per_seed.columns]
per_seed.reset_index().to_csv(OUT/'per_seed_summary.csv',index=False)
summary = pd.read_csv(OUT/'original_method_summary.csv')
compact = summary[['policy']].copy()
for metric in METRICS:
    compact[metric+'_mean'] = summary[metric+'_mean']
    compact[metric+'_scenario_std'] = summary[metric+'_std']
compact.to_csv(OUT/'benchmark.csv',index=False)
metadata=json.loads((OUT/'metadata.json').read_text())
metadata['commands']=['git clone /Users/goodday/Downloads/rohiy/SentinelRL-delivery/provenance/original-history.bundle /Users/goodday/Downloads/rohiy/SentinelRL-original-benchmark-20261026','python3 -u reproduce_original.py','python3 summarize_benchmark.py']
metadata['postprocessing']='Separate summarize_benchmark.py flattened per-seed headers and wrote compact CSV without running policies/models.'
metadata['committed_summary_cells_compared']=len(pd.read_csv(OUT/'reported_vs_reproduced.csv'))
metadata['source_or_checkpoint_patches']=[]
(OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
header=['| Original policy | Return | Critical compromised | Uncontained | Detection delay | False isolation |',
        '|---|---:|---:|---:|---:|---:|']
for _,r in summary.iterrows():
    header.append('| '+r['policy']+' | '+' | '.join(f"{r[m+'_mean']:.3f} ± {r[m+'_std']:.3f}" for m in METRICS)+' |')
readme='''# Original SentinelRL GitHub benchmark — reproduced

**Reproduction completed on2026-10-26.** This directory is isolated from the corrected/trained delivery. It restores the user's original commit `dff57ab019fde89506ea2969d7c384ce532a2bbf` using the preserved complete Git bundle. The original tracked source, checkpoint weights, environment, reward, action logic and evaluation seeds were not changed.

All three supplied checkpoints (seeds17,29,43) loaded strictly into real PyTorch2.8.0 with no missing/unexpected keys and no compatibility patch. Each checkpoint records13,440steps/1,680updates and280training episodes. This reproduction evaluates those supplied weights; it does not claim to independently reproduce their historical neural training. The original tabular model had no saved checkpoint, so the unmodified original ablation function trained it for280episodes per seed before evaluation.

The original main evaluation has30held-out scenario seeds `300000+97*i` for i0..29. For learned policies, original statistics first average the three checkpoints within each scenario, then report the sample mean and sample standard deviation across the30scenario averages. Table entries below are **mean ± scenario SD**, not confidence intervals.

'''+ '\n'.join(header)+'''

**Reproduction matches the committed results:** all 136 rounded summary mean/SD cells match exactly at the committed three-decimal precision. The original main raw metrics differ only by at most3.55×10⁻¹⁵ from CSV floating-point formatting. Full-model rows independently rerun inside the ablation are exactly identical to the main evaluation rows. There are210main raw rows and270ablation rows, or390unique policy/checkpoint/scenario rows after removing the duplicated full-model evaluation.

## Confidence and seed variation

`reproduction/original_method_summary.csv` adds10,000-replicate scenario-bootstrap95%intervals conditional on the three supplied checkpoints. The full-model mean-return interval is[−1.4034,0.4508]. It does not include uncertainty over a broader population of independently trained agents. `reproduction/per_seed_summary.csv` retains all individual seed mean/SD values. Full-model seed17/29/43 mean returns are−.534393/−.534393/−.010360; critical compromises are.133333/.133333/.066667. Tabular seed means vary substantially (−.919493/−6.502007/−1.633953), which the original scenario-averaged summary can obscure.

## What this means

This establishes faithful execution of the original GitHub benchmark. It does **not** establish that the original policy is purely learned or a fair winner against the corrected agent. The original observation exposes actual entry/infection events indirectly through belief/history and counters; its learned Q output is clamped to±.12 before a much larger handcrafted prior, and its safety shield changes action scores. It also uses different rewards and perfect decoy immunity. Original “detection” includes ground-truth-labeled alerts, unlike the corrected explicit-confirmation metric. Absolute returns and detection values therefore cannot be compared directly across the two environments.

The original NoStealth label disables the stealth detector/prior flag at inference on the same full checkpoint; it is not a separately retrained architecture. Its remaining graph/network, handcrafted prior and shield are preserved. All reported original controls are included; no optimization or tuning was performed.

## Reproduce and inspect

From a **fresh** restored clone of the same commit, copy these two reproduction helper scripts, then run:

```sh
python3 -u reproduce_original.py
python3 summarize_benchmark.py
```

The driver invokes the unmodified original `src.experiment.evaluate(episodes=30,model_seeds=(17,29,43))` and `evaluate_ablation(...)`, redirecting only output paths to `reproduction/`. It protects an existing completed reproduction from overwrite. CPU threads were set to1as an execution setting; no source/algorithm/reward/weight patch was necessary. Original tracked `results/` remain the reported evidence; `reproduction/` contains fresh measurements.

Actual runtime was5.728seconds total, including tabular training, inference, aggregation and additional bootstrap postprocessing in the driver. Python3.12.5, NumPy2.2.6, pandas2.2.2, PyTorch2.8.0, Matplotlib3.9.2 on arm64CPU. The original committed dependency snapshot listed newer versions; strict loading and numeric reproduction succeeded despite that difference.

Key files: `reproduction/benchmark.csv` (compact table), `evaluation_matrix.csv` (all390unique raw rows), `per_seed_summary.csv`, `scenario_averaged_metrics.csv`, `reported_vs_reproduced.csv`, and `metadata.json` (exact commands, versions, checkpoint hashes, configs, strict-load proof and timing). `reproduction/file-hashes.json` covers the fresh evidence and helper scripts. No GitHub push or changes to the corrected project occurred.
'''
# Readable prose spacing; preserve literal inline code and existing numbers.
replacements={'on2026':'on 2026','seeds17,29,43':'seeds 17, 29, 43','PyTorch2.8.0':'PyTorch 2.8.0','records13,440steps/1,680updates':'records 13,440 steps/1,680 updates','and280training':'and 280 training','for280episodes':'for 280 episodes','has30held-out':'has 30 held-out','i0..29':'i=0..29','the30scenario':'the 30 scenario','at most3.55':'at most 3.55','are210main':'are 210 main','and270ablation':'and 270 ablation','or390unique':'or 390 unique','adds10,000-replicate':'adds 10,000-replicate','scenario-bootstrap95%intervals':'scenario-bootstrap 95% intervals','is[−':'is [−','seed17/29/43':'seed 17/29/43','are−':'are −','are.133333':'are .133333','clamped to±':'clamped to ±','set to1as':'set to 1 as','was5.728seconds':'was 5.728 seconds','Python3.12.5':'Python 3.12.5','NumPy2.2.6':'NumPy 2.2.6','pandas2.2.2':'pandas 2.2.2','Matplotlib3.9.2':'Matplotlib 3.9.2','arm64CPU':'arm64 CPU','all390unique':'all 390 unique'}
for a,b in replacements.items():readme=readme.replace(a,b)
(ROOT/'BENCHMARK_README.md').write_text(readme)
paths=[*OUT.glob('*'),ROOT/'reproduce_original.py',ROOT/'summarize_benchmark.py',ROOT/'BENCHMARK_README.md']
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths) if p.is_file() and p.name!='file-hashes.json'}
(OUT/'file-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print('Wrote compact benchmark, flat per-seed CSV, reproduction README and file hashes; no model run.')
