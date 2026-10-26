"""Train fresh prespecified seeds using an immutable selected development runtime.

No hyperparameter override, checkpoint selection, validation or final test occurs.
The selected checkpoint supplies class/config/source identity; its trained weights
are discarded. Each new seed starts from its own standard random initialization.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from src.environment import EnvConfig,SentinelEnv
from src.policy_loading import load_snapshot_policy

ROOT=Path(__file__).resolve().parent
FINAL_SEEDS=(17,29,43,71,89)


def dump(path,payload):
    Path(path).write_text(json.dumps(payload,indent=2)+'\n')


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--seeds',type=int,nargs='+',required=True)
    args=p.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or not set(args.seeds)<=set(FINAL_SEEDS):
        raise ValueError('only unique prespecified final training seeds17,29,43,71,89 are allowed')
    template=args.template.resolve()
    prototype=load_snapshot_policy(template)
    payload=torch.load(template,map_location='cpu',weights_only=True)
    episodes=int(payload['metadata']['training_episodes'])
    if episodes<1 or payload['environment_steps']!=episodes*EnvConfig().horizon:
        raise ValueError('selected checkpoint must represent its complete fixed training budget')
    source_hashes=payload['metadata']['source_sha256']
    if hashlib.sha256((ROOT/'src/environment.py').read_bytes()).hexdigest()!=source_hashes['src/environment.py']:
        raise ValueError('current training environment differs from frozen selected runtime')
    intended=[ROOT/'candidates'/f'{args.name}_seed{seed}' for seed in args.seeds]
    if any(path.exists() for path in intended):
        raise FileExistsError('preserving existing final-run artifacts')
    template_sha=hashlib.sha256(template.read_bytes()).hexdigest()
    template_parameters=prototype.parameter_hash()
    for seed,output in zip(args.seeds,intended):
        output.mkdir(parents=True)
        source=output/'source';source.mkdir()
        frozen_hashes={}
        for name,expected in source_hashes.items():
            original=template.parent/'source'/name
            if hashlib.sha256(original.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'selected runtime changed:{name}')
            destination=source/name;destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(original.read_bytes());frozen_hashes[name]=expected
        controller=source/'train_final.py';controller.write_bytes(Path(__file__).read_bytes())
        frozen_hashes['train_final.py']=hashlib.sha256(controller.read_bytes()).hexdigest()
        # Instantiate the exact immutable implementation, rather than adapting
        # an evolving live module or reusing any development-trained weights.
        agent=type(prototype)(seed,prototype.config)
        if agent.parameter_hash()==template_parameters:
            raise RuntimeError('fresh training must not reuse development weights')
        manifest={'status':'RUNNING','family':args.name,'assigned_training_seed':seed,
                  'prespecified_training_seeds':list(FINAL_SEEDS),'training_episodes':episodes,
                  'selected_template':str(template.relative_to(ROOT)),'selected_template_sha256':template_sha,
                  'template_parameter_hash':template_parameters,'config':asdict(agent.config),
                  'environment':asdict(EnvConfig()),'source_sha256':frozen_hashes,
                  'training_scenario_rule':'1000000*training_seed+episode_index',
                  'timestamp_utc':datetime.now(timezone.utc).isoformat(),'device':'cpu',
                  'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__),
                  'validation_executed':False,'final_test_executed':False,
                  'checkpoint_selection':'fixed selected budget for every prespecified seed',
                  'preserved_checkpoint_episodes':[episodes//2,episodes],
                  'initial_parameter_hash':agent.parameter_hash()}
        dump(output/'manifest.json',manifest)
        agent.save(output/f'seed{seed}_initial.pt',{'source_sha256':frozen_hashes,'family':args.name,
                   'training_episodes':0,'checkpoint':'untrained_initial','selected_template_sha256':template_sha})
        rows=[];start=time.monotonic()
        for episode in range(episodes):
            scenario=1000000*seed+episode
            env=SentinelEnv(scenario);obs,info=env.reset(scenario);agent.reset(obs,info['action_mask'])
            update=None
            for _ in range(env.config.horizon):
                action=agent.act(obs,info['action_mask'],explore=True)
                ns,reward,done,truncated,ni=env.step(action)
                update=agent.observe(obs,action,reward,ns,done or truncated,ni['action_mask'])
                obs,info=ns,ni
                if done or truncated:break
            rows.append({'episode':episode,'training_seed':seed,'environment_seed':scenario,**env.metrics(),
                         'epsilon':agent.epsilon(),'environment_steps':agent.environment_steps,**(update or {}),
                         'elapsed_seconds':time.monotonic()-start})
            if episode+1==episodes//2:
                agent.save(output/f'seed{seed}_ep{episode+1}.pt',
                           {'source_sha256':frozen_hashes,'family':args.name,'training_episodes':episode+1,
                            'selected_template_sha256':template_sha,'checkpoint_selection':'fixed training midpoint preserved for whole family',
                            'validation_executed':False,'final_test_executed':False})
            if (episode+1)%500==0 or episode+1==episodes:
                write_csv(output/f'training_seed{seed}.csv',rows)
                print(f'{args.name} seed={seed} episode={episode+1}/{episodes} return500={np.mean([r["return"] for r in rows[-500:]]):.4f} updates={agent.updates} seconds={time.monotonic()-start:.1f}',flush=True)
        checkpoint=output/f'seed{seed}_ep{episodes}.pt'
        agent.save(checkpoint,{'source_sha256':frozen_hashes,'family':args.name,'training_episodes':episodes,
                              'selected_template_sha256':template_sha,'checkpoint_selection':'fixed final budget',
                              'validation_executed':False,'final_test_executed':False})
        manifest.update(status='COMPLETE',elapsed_seconds=time.monotonic()-start,
                        environment_steps=agent.environment_steps,updates=agent.updates,
                        decision_transitions=agent.decision_transitions,final_parameter_hash=agent.parameter_hash(),
                        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        if agent.parameter_hash()==agent.initial_parameter_hash or agent.environment_steps!=episodes*48:
            raise RuntimeError('final training evidence does not match fixed protocol')
        dump(output/'manifest.json',manifest)

if __name__=='__main__':main()
