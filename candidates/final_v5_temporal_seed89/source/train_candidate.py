"""Development-only candidate training. Does not run reserved final test scenarios."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import platform
import time
import numpy as np
import torch
from src.champion import ChampionAgent, ChampionConfig
from src.environment import SentinelEnv, EnvConfig
from src.policies import AlertResponderPolicy

ROOT = Path(__file__).resolve().parent
METRICS = ('return','total_compromises','critical_compromises','uncontained_at_end','false_isolations','successful_isolations','action_costs','invalid_actions','confirmed_fraction','censored_confirmation_delay')


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def evaluate(policy, scenarios=200):
    rows = []
    for i in range(scenarios):
        seed = 600000 + 101*i
        env = SentinelEnv(seed); obs, info = env.reset(seed)
        if hasattr(policy, 'reset'): policy.reset(obs, info['action_mask'])
        for _ in range(env.config.horizon):
            action = policy.act(obs, info['action_mask'], explore=False)
            obs, _, done, _, info = env.step(action)
            if done: break
        rows.append({'scenario_index':i,'environment_seed':seed,**env.metrics(),
                     **{f'action_{a}':int(c) for a,c in enumerate(env.action_counts)}})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True); p.add_argument('--seeds',type=int,nargs='+',default=[101])
    p.add_argument('--episodes',type=int,default=20000)
    p.add_argument('--checkpoints',type=int,nargs='+',default=[2000,5000,10000,20000])
    p.add_argument('--validation',type=int,default=200); p.add_argument('--config',default='{}')
    args = p.parse_args()
    config = ChampionConfig(**json.loads(args.config))
    output = ROOT/'candidates'/args.name
    output.mkdir(parents=True,exist_ok=False)
    source = output/'source'; source.mkdir()
    paths = ['src/environment.py','src/belief.py','src/champion.py','train_candidate.py']
    hashes = {}
    for name in paths:
        target=source/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes((ROOT/name).read_bytes())
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest={'name':args.name,'created_utc':datetime.now(timezone.utc).isoformat(),'status':'RUNNING',
              'training_seeds':args.seeds,'training_episodes':args.episodes,'config':asdict(config),
              'environment':asdict(EnvConfig()),'source_sha256':hashes,'python':platform.python_version(),
              'torch':str(torch.__version__),'numpy':np.__version__,'device':'cpu',
              'training_scenario_rule':'1000000 * training_seed + episode_index',
              'validation_scenario_rule':'600000 + 101 * scenario_index','validation_scenarios':args.validation,
              'test_scenarios_used':False,'forced_wait_compression':'actual suffix rewards folded into prior transition with duration discount'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summaries=[]
    baseline_rows=evaluate(AlertResponderPolicy(),args.validation)
    write_csv(output/'validation_alert_rule.csv',baseline_rows)
    summaries.append({'policy':'alert_rule','seed':0,'episodes':0,**{m:float(np.mean([r[m] for r in baseline_rows])) for m in METRICS}})
    for seed in args.seeds:
        agent=ChampionAgent(seed,config)
        agent.save(output/f'seed{seed}_initial.pt',{'checkpoint':'initial','candidate':args.name})
        rows=[]; start=time.monotonic()
        for episode in range(args.episodes):
            scenario=1000000*seed+episode
            env=SentinelEnv(scenario); obs,info=env.reset(scenario); agent.reset(obs,info['action_mask'])
            update=None
            for _ in range(env.config.horizon):
                action=agent.act(obs,info['action_mask'],explore=True)
                ns,r,done,truncated,ni=env.step(action)
                update=agent.observe(obs,action,r,ns,done or truncated,ni['action_mask'])
                obs,info=ns,ni
                if done or truncated: break
            rows.append({'episode':episode,'training_seed':seed,'environment_seed':scenario,**env.metrics(),
                         'epsilon':agent.epsilon(),'environment_steps':agent.environment_steps,**(update or {}),
                         'elapsed_seconds':time.monotonic()-start})
            if (episode+1)%500==0 or episode+1==args.episodes:
                write_csv(output/f'training_seed{seed}.csv',rows)
                print(f'{args.name} seed={seed} episode={episode+1} return500={np.mean([r["return"] for r in rows[-500:]]):.3f} updates={agent.updates} seconds={time.monotonic()-start:.1f}',flush=True)
            if episode+1 in args.checkpoints or episode+1==args.episodes:
                checkpoint=output/f'seed{seed}_ep{episode+1}.pt'
                agent.save(checkpoint,{'candidate':args.name,'training_episodes':episode+1,'source_sha256':hashes,'validation_scenarios':args.validation,'test_scenarios_used':False})
                validation=evaluate(agent,args.validation)
                write_csv(output/f'validation_seed{seed}_ep{episode+1}.csv',validation)
                summary={'policy':'champion','seed':seed,'episodes':episode+1,**{m:float(np.mean([r[m] for r in validation])) for m in METRICS}}
                summaries.append(summary); write_csv(output/'validation_summary.csv',summaries)
                print('VALIDATION '+json.dumps(summary),flush=True)
    manifest['status']='COMPLETE'; manifest['elapsed_seconds']=time.monotonic()-start
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': main()
