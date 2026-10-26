#!/usr/bin/env python3
"""Common-environment paired benchmark. Final test requires explicit audit authorization."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
import shutil
import numpy as np
import torch
from src.environment import EnvConfig, SentinelEnv
from src.comparators import PRETRAINED_POLICIES, comparator_seeds, make_comparator, source_hash, BeliefThresholdPolicy
from src.policy_loading import load_snapshot_policy
from src.ensembles import LearnedQEnsemble, OriginalQEnsemble, PreviousQEnsemble

ROOT = Path(__file__).resolve().parent


def scenario_seeds(split, episodes, specification=None):
    if split == 'validation':
        return [600000 + 101*i for i in range(episodes)]
    if split == 'final':
        authorization = ROOT/'provenance/FINAL_TEST_AUTHORIZATION.json'
        if not authorization.exists() or not json.loads(authorization.read_text()).get('authorized'):
            raise PermissionError('Final scenarios are sealed until the parent review explicitly authorizes the frozen configuration')
        if specification is None:
            raise ValueError('final test requires a frozen comparison specification')
        approved=json.loads(authorization.read_text())
        if approved.get('protocol_sha256') != specification['_file_sha256']:
            raise PermissionError('authorization does not match the frozen protocol bytes')
        if episodes != specification['final_scenarios'] or episodes != approved.get('scenario_count'):
            raise ValueError('final scenario count differs from the approved frozen protocol')
        return [900000 + 103*i for i in range(episodes)]
    raise ValueError(split)


def load_spec_policy(spec, model_seed, scenario, configuration):
    kind=spec['kind']
    if kind=='checkpoint':
        checkpoint=ROOT/spec['checkpoints'][str(model_seed)]
        expected=spec['checkpoint_sha256'][str(model_seed)]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=expected:
            raise ValueError('frozen checkpoint bytes changed:'+str(checkpoint))
        return load_snapshot_policy(checkpoint)
    if kind=='ensemble':
        subtype=spec['subtype']
        if subtype=='snapshot_learned':
            for member_seed,path in spec['checkpoints'].items():
                if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=spec['checkpoint_sha256'][member_seed]:
                    raise ValueError('frozen ensemble member bytes changed')
            return LearnedQEnsemble([ROOT/path for path in spec['checkpoints'].values()])
        if subtype=='original':
            return OriginalQEnsemble(tuple(spec.get('member_seeds',[17,29,43])),spec.get('no_stealth',False))
        if subtype=='previous':
            return PreviousQEnsemble(spec.get('comparator','previous_ddqn'),tuple(spec.get('member_seeds',[17,29,43])))
        raise ValueError('unknown ensemble subtype')
    if kind=='threshold':
        return BeliefThresholdPolicy(**spec['configuration'])
    if kind=='no_op':
        from src.policies import NoOpPolicy
        return NoOpPolicy()
    if kind=='comparator':
        # Distinct policy RNG stream; no environment variate is reused as action noise.
        return make_comparator(spec['comparator'],model_seed,scenario,configuration)
    raise ValueError(kind)


def run_episode(policy, scenario, config):
    env = SentinelEnv(seed=scenario, config=config)
    observation, info = env.reset(scenario)
    if hasattr(policy,'reset'):
        policy.reset(observation, info['action_mask'])
    for _ in range(config.horizon):
        action = policy.act(observation, info['action_mask'], explore=False)
        observation, _, done, truncated, info = env.step(action)
        if done or truncated:
            break
    return {**env.metrics(), **{f'action_{a}':int(v) for a,v in enumerate(env.action_counts)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=('validation','final'),default='validation')
    parser.add_argument('--episodes',type=int)
    parser.add_argument('--spec',type=Path,help='Frozen policy/checkpoint specification for integrated validation or final test')
    parser.add_argument('--policies',nargs='+',choices=PRETRAINED_POLICIES,default=list(PRETRAINED_POLICIES))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    specification=None
    if args.spec:
        specification=json.loads(args.spec.read_text())
        specification['_file_sha256']=hashlib.sha256(args.spec.read_bytes()).hexdigest()
        for relative,expected in specification.get('runtime_hashes',{}).items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
                raise ValueError('frozen runtime changed:'+relative)
        env_hash=hashlib.sha256((ROOT/'src/environment.py').read_bytes()).hexdigest()
        if specification.get('environment_sha256')!=env_hash or specification.get('environment')!=asdict(EnvConfig()):
            raise ValueError('frozen specification does not match unchanged environment')
    if args.episodes is None:
        args.episodes=specification['final_scenarios'] if args.split=='final' and specification else 200
    if args.episodes < 1:
        parser.error('episodes must be positive')
    scenarios = scenario_seeds(args.split,args.episodes,specification)
    if args.output.exists():
        raise FileExistsError('preserving existing benchmark evidence; choose a new output directory')
    args.output.mkdir(parents=True)
    config = EnvConfig()
    manifest={'started_utc':datetime.now(timezone.utc).isoformat(),'split':args.split,'episodes':args.episodes,
              'scenario_seeds':scenarios,'environment':asdict(config),
              'environment_sha256':hashlib.sha256((ROOT/'src/environment.py').read_bytes()).hexdigest(),
              'source_sha256':source_hash(ROOT/'src'),'policies':args.policies,'status':'RUNNING',
              'torch':str(torch.__version__),'numpy':np.__version__,'frozen_specification':specification}
    source_dir=args.output/'source'
    for relative in ('benchmark.py','src/comparators.py','src/policy_loading.py','src/belief.py','src/environment.py'):
        target=source_dir/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    start=time.monotonic();rows=[]
    policy_specs=specification['policies'] if specification else [
        {'name':kind,'kind':'comparator','comparator':kind,'model_seeds':list(comparator_seeds(kind))} for kind in args.policies]
    for policy_spec in policy_specs:
        kind=policy_spec['name'];policy_start=time.monotonic()
        model_seeds=list(map(int,policy_spec['checkpoints'])) if policy_spec['kind']=='checkpoint' else policy_spec.get('model_seeds',[0])
        for seed in model_seeds:
            policy=None
            for scenario in scenarios:
                if policy is None or policy_spec.get('comparator')=='original_random':
                    policy=load_spec_policy(policy_spec,seed,scenario,config)
                row={'policy':kind,'model_seed':seed,'environment_seed':scenario,
                     **run_episode(policy,scenario,config)}
                rows.append(row)
        with (args.output/'episodes.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        values=[r for r in rows if r['policy']==kind]
        print(f"{kind}: n={len(values)} return={np.mean([r['return'] for r in values]):.5f} critical={np.mean([r['critical_compromises'] for r in values]):.5f} elapsed={time.monotonic()-policy_start:.2f}s",flush=True)
    metrics=list(SentinelEnv(seed=1).metrics())
    summary=[]
    for policy_spec in policy_specs:
        kind=policy_spec['name']
        selected=[r for r in rows if r['policy']==kind]
        summary.append({'policy':kind,'model_seeds':len({r['model_seed'] for r in selected}),'scenarios':len(scenarios),
                        **{metric+'_mean':float(np.mean([r[metric] for r in selected])) for metric in metrics}})
    with (args.output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    manifest.update({'status':'COMPLETE','elapsed_seconds':time.monotonic()-start,'rows':len(rows)})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
