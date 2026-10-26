"""Spawned immutable policy/seed/scenario chunks for the common benchmark."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,time
import torch


def evaluate_chunk(task):
    from benchmark import ROOT,scenario_seeds,load_spec_policy,run_episode
    from src.environment import EnvConfig
    torch.set_num_threads(1)
    specification=task['specification']
    if specification:
        if hashlib.sha256(Path(task['spec_file']).read_bytes()).hexdigest()!=specification['_file_sha256']:
            raise ValueError('worker protocol bytes differ')
        for name,expected in specification.get('runtime_hashes',{}).items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('worker runtime differs:'+name)
    complete=scenario_seeds(task['split'],task['episodes'],specification)
    scenarios=complete[task['first']:task['stop']]
    path=Path(task['chunk_path'])
    if path.exists():raise FileExistsError('immutable chunk already exists')
    start=time.monotonic();rows=[];spec=task['policy'];config=EnvConfig()
    for seed in task['model_seeds']:
        policy=None
        for scenario in scenarios:
            if policy is None or spec.get('comparator')=='original_random':policy=load_spec_policy(spec,seed,scenario,config)
            rows.append({'policy':spec['name'],'model_seed':seed,'environment_seed':scenario,**run_episode(policy,scenario,config)})
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    record={'policy':spec['name'],'model_seeds':task['model_seeds'],'scenario_first_index':task['first'],
            'scenario_stop_index':task['stop'],'rows':len(rows),'elapsed_seconds':time.monotonic()-start,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'created_utc':datetime.now(timezone.utc).isoformat()}
    path.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    return record
