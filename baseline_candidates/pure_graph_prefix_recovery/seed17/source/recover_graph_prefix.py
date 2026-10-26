#!/usr/bin/env python3
"""Recover missing10k snapshots by exact deterministic prefix replay, never relabel20k."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,shutil,time
import numpy as np
from src.environment import SentinelEnv
from src.policy_loading import load_snapshot_policy

ROOT=Path(__file__).resolve().parent
RUNS={17:ROOT/'baseline_candidates/final_graph_pure_a',71:ROOT/'baseline_candidates/final_graph_pure_b'}
OUTPUT=ROOT/'baseline_candidates/pure_graph_prefix_recovery'
OUTPUT.mkdir(parents=True,exist_ok=False)
records=[]
for seed,original in RUNS.items():
 checkpoint=original/f'seed{seed}_ep20000.pt'
 original_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 template=load_snapshot_policy(checkpoint)
 agent=type(template)(seed,config=template.config,env_config=template.env_config,kind=template.kind)
 if agent.parameter_hash()!=template.initial_parameter_hash:raise ValueError('initial parameters differ from original run')
 folder=OUTPUT/f'seed{seed}';folder.mkdir()
 shutil.copytree(original/'source',folder/'source')
 shutil.copy2(__file__,folder/'source/recover_graph_prefix.py')
 hashes={**template.checkpoint_metadata['source_sha256'],'recover_graph_prefix.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 expected=list(csv.DictReader((original/f'training_seed{seed}.csv').open()))[:10000]
 if len(expected)!=10000:raise ValueError('complete original prefix is required')
 rows=[];start=time.monotonic();maximum_difference=0.
 for episode in range(10000):
  scenario=1000000*seed+episode
  env=SentinelEnv(scenario);obs,info=env.reset(scenario);agent.reset(obs,info['action_mask'])
  update=None
  for _ in range(env.config.horizon):
   action=agent.act(obs,info['action_mask'],explore=True)
   ns,reward,done,truncated,ni=env.step(action)
   update=agent.observe(obs,action,reward,ns,done or truncated,ni['action_mask'])
   obs,info=ns,ni
   if done or truncated:break
  row={'episode':episode,'training_seed':seed,'environment_seed':scenario,**env.metrics(),
       'epsilon':agent.epsilon(),'environment_steps':agent.environment_steps,**(update or {}),
       'elapsed_seconds':time.monotonic()-start}
  for key,value in row.items():
   if key=='elapsed_seconds':continue
   delta=abs(float(value)-float(expected[episode][key]));maximum_difference=max(maximum_difference,delta)
   if delta>1e-9:raise ValueError(f'prefix differs at seed{seed},episode{episode},{key}:{value} vs {expected[episode][key]}')
  rows.append(row)
  if (episode+1)%500==0:
   with (folder/'training_prefix.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
   print(f'prefix recovery seed={seed} episode={episode+1}/10000 all recorded numeric fields match;seconds={time.monotonic()-start:.1f}',flush=True)
 metadata={'checkpoint_origin':'deterministic_prefix_replay','created_utc':datetime.now(timezone.utc).isoformat(),
           'source_sha256':hashes,'training_episodes':10000,'original_training_budget':20000,
           'original20k_checkpoint_sha256':original_sha,'prefix_raw_fields_match':True,'excluded_field':'elapsed_seconds',
           'no_reference10k_parameter_hash_existed':True,'test_scenarios_used':False}
 recovered=folder/f'seed{seed}_ep10000.pt';agent.save(recovered,metadata)
 assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==original_sha
 record={'seed':seed,'episodes_replayed':10000,'actual_recovery_environment_steps':agent.environment_steps,
         'actual_recovery_gradient_updates':agent.updates,'elapsed_seconds':time.monotonic()-start,
         'original20k_checkpoint_unchanged':True,'initial_parameter_hash_match':True,
         'all10000raw_rows_all_numeric_fields_match':True,'maximum_numeric_difference':maximum_difference,
         'excluded_field':'elapsed_seconds','recovered_checkpoint_sha256':hashlib.sha256(recovered.read_bytes()).hexdigest(),
         'no_original10k_reference_hash_existed':True}
 (folder/'RECOVERY_PROOF.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record)
(OUTPUT/'manifest.json').write_text(json.dumps({'status':'COMPLETE','purpose':'missing10k checkpoint retention only; duplicate compute separately counted','records':records,'test_scenarios_used':False},indent=2)+'\n')
print('PREFIX RECOVERY COMPLETE',json.dumps(records),flush=True)
