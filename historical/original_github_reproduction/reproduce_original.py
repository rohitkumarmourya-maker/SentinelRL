#!/usr/bin/env python3
"""Reproduce original commit's30-scenario benchmark without changing its source."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
import matplotlib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reproduction'
SEEDS = (17, 29, 43)
METRICS = ['episode_return', 'critical_compromises', 'active_compromised_at_end', 'mean_detection_delay', 'false_isolations']


def parameter_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def bootstrap_mean(values, seed=20261026, repetitions=10000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    draws = np.array([values[rng.integers(len(values), size=len(values))].mean() for _ in range(repetitions)])
    return np.quantile(draws, [.025, .975]).tolist()


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'metadata.json').exists():
        raise FileExistsError('Preserving completed reproduction; use a new clone/directory')
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == 'dff57ab019fde89506ea2969d7c384ce532a2bbf'
    torch.set_num_threads(1)  # execution-only CPU setting; algorithm/weights/reward unchanged
    from src import experiment
    from src.dqn import DQNAgent
    assert experiment.torch is torch and DQNAgent.__module__ == 'src.dqn'
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    loads = []
    for seed in SEEDS:
        path = ROOT / 'models' / f'sentinel_dqn_seed{seed}.pt'
        agent = DQNAgent.load(str(path), device='cpu')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        strict = agent.online.load_state_dict(checkpoint['online'], strict=True)
        assert not strict.missing_keys and not strict.unexpected_keys
        loads.append({'seed': seed, 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'strict_load': True, 'missing_keys': [], 'unexpected_keys': [],
                      'online_parameter_hash': parameter_hash(agent.online),
                      'steps': agent.steps, 'updates': agent.updates,
                      'config': checkpoint['config'], 'metadata': checkpoint.get('metadata', {})})
    # Only paths are redirected. These are the unmodified original functions.
    experiment.RESULTS = OUT
    experiment.MODELS = ROOT / 'models'
    print('Running original evaluate(episodes=30,model_seeds=(17,29,43))', flush=True)
    main_rows = experiment.evaluate(episodes=30, model_seeds=SEEDS)
    evaluation_seconds = time.monotonic() - start
    print(f'Original main evaluation done: {len(main_rows)} rows, {evaluation_seconds:.3f}s', flush=True)
    ablation_start = time.monotonic()
    print('Running original evaluate_ablation: NoStealth toggle + three original280-episode tabular runs', flush=True)
    ablation_rows = experiment.evaluate_ablation(episodes=30, model_seeds=SEEDS)
    ablation_rows.to_csv(OUT / 'ablation_episodes.csv', index=False)
    ablation_seconds = time.monotonic() - ablation_start
    full = ablation_rows[ablation_rows.policy == 'Graph-Dueling DQN (Full)'].sort_values(['model_seed', 'eval_seed'])
    primary = main_rows[main_rows.policy == 'DQN'].sort_values(['model_seed', 'eval_seed'])
    assert np.array_equal(full[METRICS].to_numpy(), primary[METRICS].to_numpy())
    extra = ablation_rows[ablation_rows.policy != 'Graph-Dueling DQN (Full)']
    matrix = pd.concat([main_rows.assign(policy=main_rows.policy.replace({'DQN':'Graph-Dueling DQN (Full)'})), extra], ignore_index=True)
    matrix.to_csv(OUT / 'evaluation_matrix.csv', index=False)
    assert len(matrix) == 390
    # Reproduce original aggregation: average three learned seeds within scenario,
    # then sample mean/SD across30 scenarios. Also retain individual seed means.
    scenario = matrix.groupby(['policy','eval_seed'])[METRICS].mean().reset_index()
    scenario.to_csv(OUT / 'scenario_averaged_metrics.csv', index=False)
    summary_rows = []
    for policy, rows in scenario.groupby('policy'):
        result = {'policy':policy, 'scenarios':len(rows),
                  'model_seeds':3 if policy in ('Graph-Dueling DQN (Full)','Dueling DQN (No Stealth Prior)','Tabular Q-Learning') else 0}
        for metric in METRICS:
            values = rows[metric].to_numpy()
            lo,hi = bootstrap_mean(values)
            result.update({metric+'_mean':float(values.mean()), metric+'_std':float(values.std(ddof=1)),
                           metric+'_scenario_bootstrap95_low':lo, metric+'_scenario_bootstrap95_high':hi})
        summary_rows.append(result)
    pd.DataFrame(summary_rows).to_csv(OUT / 'original_method_summary.csv', index=False)
    matrix.groupby(['policy','model_seed'])[METRICS].agg(['mean','std']).to_csv(OUT / 'per_seed_metrics.csv')
    comparisons=[]
    for filename in ('summary.csv','ablation_summary.csv'):
        reported=pd.read_csv(ROOT/'results'/filename)
        actual=pd.read_csv(OUT/filename)
        shutil.copy2(ROOT/'results'/filename, OUT/('reported_'+filename))
        for _, row in reported.iterrows():
            actual_row=actual[actual.policy==row.policy].iloc[0]
            for key in reported.columns:
                if key=='policy':continue
                old,new=float(row[key]),float(actual_row[key])
                comparisons.append({'table':filename,'policy':row.policy,'metric':key,
                                    'reported':old,'reproduced':new,'difference':new-old,
                                    'match_at_committed_3_decimals':bool(np.isclose(old,new,rtol=0,atol=1e-12))})
    pd.DataFrame(comparisons).to_csv(OUT/'reported_vs_reproduced.csv',index=False)
    # Original main raw episodes exist; compare without rounding.
    reported_raw=pd.read_csv(ROOT/'results/evaluation_episodes.csv').sort_values(['policy','model_seed','eval_seed'])
    actual_raw=main_rows.sort_values(['policy','model_seed','eval_seed'])
    raw_delta=float(np.max(np.abs(reported_raw[METRICS].to_numpy()-actual_raw[METRICS].to_numpy())))
    for item in loads:
        path=ROOT/'models'/f"sentinel_dqn_seed{item['seed']}.pt"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['checkpoint_sha256']
    tracked_diff=subprocess.check_output(['git','diff','--stat','HEAD'],cwd=ROOT,text=True)
    assert not tracked_diff
    metadata={'status':'COMPLETE','original_commit':'dff57ab019fde89506ea2969d7c384ce532a2bbf',
              'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds':time.monotonic()-start,'main_evaluation_seconds':evaluation_seconds,
              'ablation_with_tabular_training_seconds':ablation_seconds,
              'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
              'torch':str(torch.__version__),'matplotlib':matplotlib.__version__,'real_pytorch':True,
              'device':'cpu','threads':1,'architecture':platform.machine(),
              'checkpoint_loads':loads,'execution_patch':'none','tracked_source_or_weights_changed':False,
              'original_heldout_seed_formula':'300000+97*i;i=0..29','dqn_checkpoint_scenarios':90,
              'main_raw_rows':len(main_rows),'ablation_raw_rows':len(ablation_rows),'unique_matrix_rows':len(matrix),
              'tabular_training':'original evaluate_ablation function:3seeds×280episodes×48steps;alpha.18,gamma.97,epsilon=max(.05,.85*(1-episode/220))',
              'all_committed_summary_cells_match':all(r['match_at_committed_3_decimals'] for r in comparisons),
              'main_raw_max_absolute_metric_difference':raw_delta,
              'ci_scope':'Scenario bootstrap conditional on the3 fixed supplied checkpoints; not population training-seed uncertainty',
              'scientific_scope':'Original reproduction only; hidden-state leakage, handcrafted Qprior/shield and different reward/env preclude fair direct comparison with corrected agent.'}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:metadata[k] for k in ('status','elapsed_seconds','main_raw_rows','ablation_raw_rows','unique_matrix_rows','all_committed_summary_cells_match','main_raw_max_absolute_metric_difference')},indent=2),flush=True)
    print(pd.DataFrame(summary_rows)[['policy',*[metric+'_mean' for metric in METRICS]]].to_string(index=False),flush=True)


if __name__=='__main__':main()
