#!/usr/bin/env python3
"""Original tabular Q-learning: exact280episode historical training or lawful20k retrain."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,shutil,time
import numpy as np
from src.original_tabular import save_original_tabular
from src.comparators import Lawful58Adapter,verify_original_source
from src.environment import SentinelEnv as CommonEnv
from vendor.original_v1.environment import SentinelEnv as HistoricalEnv
from vendor.original_v1.policies import TabularQLearningAgent

ROOT=Path(__file__).resolve().parent


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('historical','lawful'),required=True)
 p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
 args=p.parse_args();verify_original_source()
 if args.output.exists():raise FileExistsError('preserve previous tabular evidence')
 args.output.mkdir(parents=True)
 files=['src/environment.py','src/belief.py','src/comparators.py','src/original_tabular.py',
        'vendor/original_v1/policies.py','vendor/original_v1/environment.py','train_original_tabular.py']
 hashes={}
 for name in files:
  target=args.output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
  hashes[name]=hashlib.sha256(target.read_bytes()).hexdigest()
 episodes=280 if args.mode=='historical' else 20000
 manifest={'status':'RUNNING','mode':args.mode,'seeds':args.seeds,'episodes':episodes,'source_sha256':hashes,
           'alpha':.18,'gamma':.97,'epsilon_rule':'max(.05,.85*(1-episode/(episodes*220/280)))',
           'training_seed_rule':'seed*10000+episode' if args.mode=='historical' else 'seed*1000000+episode',
           'final_test_used':False,'started_utc':datetime.now(timezone.utc).isoformat()}
 (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 records=[];overall=time.monotonic()
 for seed in args.seeds:
  agent=TabularQLearningAgent(seed=seed);rows=[];start=time.monotonic();updates=0
  adapter=Lawful58Adapter() if args.mode=='lawful' else None
  for episode in range(episodes):
   scenario=seed*(10000 if args.mode=='historical' else 1000000)+episode
   env=(HistoricalEnv if args.mode=='historical' else CommonEnv)(seed=scenario)
   obs,info=env.reset(scenario)
   features=adapter.reset(obs) if adapter else obs
   mask=adapter.legacy_mask(obs,info['action_mask']) if adapter else info['action_mask']
   eps=max(.05,.85*(1-episode/(episodes*220/280)))
   for _ in range(48):
    action=agent.act(features,mask,epsilon=eps)
    nxt,reward,done,truncated,ni=env.step(action)
    if adapter:
     adapter.previous_action=action;next_features=adapter.advance(nxt)
     next_mask=adapter.legacy_mask(nxt,ni['action_mask'])
    else:next_features,next_mask=nxt,ni['action_mask']
    agent.update(features,action,reward,next_features,done or truncated,next_mask);updates+=1
    obs,features,mask=nxt,next_features,next_mask
    if done or truncated:break
   metrics=env.summary() if args.mode=='historical' else env.metrics()
   rows.append({'episode':episode,'training_seed':seed,'environment_seed':scenario,**metrics,'epsilon':eps,'updates':updates,'elapsed_seconds':time.monotonic()-start})
   if (episode+1)%1000==0 or episode+1==episodes:
    with (args.output/f'training_seed{seed}.csv').open('w',newline='') as f:
     w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'original tabular {args.mode} seed={seed} episodes={episode+1}/{episodes} seconds={time.monotonic()-start:.1f}',flush=True)
   if episode+1 in ({280} if args.mode=='historical' else {10000,20000}):
    metadata={'mode':args.mode,'seed':seed,'episodes':episode+1,'environment_steps':updates,'source_sha256':hashes,'test_scenarios_used':False,'created_utc':datetime.now(timezone.utc).isoformat()}
    save_original_tabular(agent,args.output/f'seed{seed}_ep{episode+1}.json',metadata)
  record={'seed':seed,'episodes':episodes,'environment_steps':updates,'elapsed_seconds':time.monotonic()-start,'states':len(agent.q_table)}
  records.append(record)
  if args.mode=='historical':
   evaluation=[]
   for i in range(30):
    scenario=300000+97*i;env=HistoricalEnv(scenario);obs,info=env.reset(scenario)
    for _ in range(48):
     action=agent.act(obs,info['action_mask']);obs,_,done,_,info=env.step(action)
     if done:break
    evaluation.append({'seed':seed,'eval_seed':scenario,**env.summary()})
   with (args.output/f'historical_check_seed{seed}.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(evaluation[0]));w.writeheader();w.writerows(evaluation)
 manifest.update({'status':'COMPLETE','elapsed_seconds':time.monotonic()-overall,'records':records})
 (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
