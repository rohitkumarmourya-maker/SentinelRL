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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from src.environment import EnvConfig, SentinelEnv
from src.comparators import PRETRAINED_POLICIES, comparator_seeds, make_comparator, source_hash, BeliefThresholdPolicy
from src.policy_loading import load_snapshot_policy,checkpoint_members
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
        if specification.get('final_seed_base')!=900000000 or specification.get('final_seed_stride')!=103 or episodes!=10000:
            raise ValueError('final scenario amendment requires900million base,stride103,count10000')
        return [900000000 + 103*i for i in range(episodes)]
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
            members=checkpoint_members(spec)
            for member in members:
                if hashlib.sha256((ROOT/member['checkpoint']).read_bytes()).hexdigest()!=spec['checkpoint_sha256'][member['member_id']]:
                    raise ValueError('frozen ensemble member bytes changed')
            return LearnedQEnsemble([ROOT/member['checkpoint'] for member in members])
        if subtype=='original':
            return OriginalQEnsemble(tuple(spec.get('member_seeds',[17,29,43])),spec.get('no_stealth',False))
        if subtype=='previous':
            return PreviousQEnsemble(spec.get('comparator','previous_ddqn'),tuple(spec.get('member_seeds',[17,29,43])))
        raise ValueError('unknown ensemble subtype')
    if kind=='tabular_checkpoint':
        from src.original_tabular import OriginalTabularPolicy
        checkpoint=ROOT/spec['checkpoints'][str(model_seed)]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=spec['checkpoint_sha256'][str(model_seed)]:
            raise ValueError('frozen original tabular bytes changed')
        return OriginalTabularPolicy(checkpoint)
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
    actions=[];rewards=[]
    for _ in range(config.horizon):
        action = policy.act(observation, info['action_mask'], explore=False)
        observation, reward, done, truncated, info = env.step(action)
        actions.append(int(action));rewards.append(float(reward))
        if done or truncated:
            break
    return {**env.metrics(), **{f'action_{a}':int(v) for a,v in enumerate(env.action_counts)},
            'action_sequence':json.dumps(actions,separators=(',',':')),
            'reward_sequence':json.dumps(rewards,separators=(',',':'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=('validation','final'),default='validation')
    parser.add_argument('--episodes',type=int)
    parser.add_argument('--spec',type=Path,help='Frozen policy/checkpoint specification for integrated validation or final test')
    parser.add_argument('--policies',nargs='+',choices=PRETRAINED_POLICIES,default=list(PRETRAINED_POLICIES))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--chunk-size',type=int,default=1000)
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
    if not 1<=args.workers<=8 or args.chunk_size<1:
        parser.error('workers must be1..8 and chunk sizepositive')
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
    chunks=args.output/'chunks';chunks.mkdir()
    tasks=[];expected=set()
    for policy_index,policy_spec in enumerate(policy_specs):
        kind=policy_spec['name']
        model_seeds=list(map(int,policy_spec['checkpoints'])) if policy_spec['kind'] in ('checkpoint','tabular_checkpoint') else policy_spec.get('model_seeds',[0])
        for seed in model_seeds:
            for first in range(0,len(scenarios),args.chunk_size):
                stop=min(first+args.chunk_size,len(scenarios))
                tasks.append({'policy':policy_spec,'model_seeds':[seed],'first':first,'stop':stop,
                  'episodes':args.episodes,'split':args.split,'specification':specification,
                  'spec_file':str(args.spec.resolve()) if args.spec else None,
                  'chunk_path':str((chunks/f'p{policy_index:02d}_seed{seed}_{first:05d}.csv').resolve())})
                expected.update((kind,seed,scenario) for scenario in scenarios[first:stop])
    from benchmark_worker import evaluate_chunk
    records=[]
    if args.workers==1:
        for task in tasks:records.append(evaluate_chunk(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as executor:
            futures={executor.submit(evaluate_chunk,task):task for task in tasks}
            for future in as_completed(futures):
                record=future.result();records.append(record)
                print(f"completed {len(records)}/{len(tasks)} chunks: {record['policy']} seed={record['model_seeds']} rows={record['rows']}",flush=True)
    for task in tasks:
        with Path(task['chunk_path']).open() as f:
            for row in csv.DictReader(f):
                for key,value in list(row.items()):
                    if key in ('policy','action_sequence','reward_sequence'):continue
                    row[key]=float(value) if key not in ('model_seed','environment_seed') and not key.startswith('action_') else int(value)
                rows.append(row)
    actual=[(r['policy'],int(r['model_seed']),int(r['environment_seed'])) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('incomplete or duplicated frozen coverage')
    with (args.output/'episodes.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest['worker_chunks']=records;manifest['workers']=args.workers;manifest['coverage_unique_complete']=True
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
