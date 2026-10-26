"""Frozen all-member development sanity only; cannot run sealed final scenarios."""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import asdict
import csv,hashlib,json
import numpy as np
import torch
from src.environment import EnvConfig
from src.ensembles import LearnedQEnsemble
from src.policy_loading import load_snapshot_policy
from train_candidate import evaluate,write_csv,METRICS

ROOT=Path(__file__).resolve().parent
SEEDS=(17,29,43,71,89)
TIMES=(10000,20000)


def main():
    family_path=ROOT/'candidates/final_v5_temporal_family.json'
    family=json.loads(family_path.read_text())
    if not family['algorithm_configuration_deployment_frozen']:
        raise ValueError('algorithm/deployment must already be frozen')
    output=ROOT/'validation/final_v5_temporal_sanity'
    if output.exists():raise FileExistsError('preserving development evidence')
    paths=[];hashes={};training_evidence={}
    for seed in SEEDS:
        directory=ROOT/f'candidates/final_v5_temporal_seed{seed}'
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest['status']!='COMPLETE':raise ValueError(f'seed{seed} training is incomplete')
        assert manifest['config']==family['configuration']
        assert manifest['source_sha256']==family['source_sha256_common']
        for episode in TIMES:
            checkpoint=directory/f'seed{seed}_ep{episode}.pt'
            payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert payload['seed']==seed and payload['environment_steps']==episode*48
            assert payload['metadata']['training_episodes']==episode
            assert payload['config']==family['configuration']
            assert payload['metadata']['source_sha256']==family['source_sha256_common']
            assert payload['environment']==asdict(EnvConfig())
            assert payload['updates']>0 and payload['parameter_hash']!=payload['initial_parameter_hash']
            paths.append(checkpoint);hashes[str(checkpoint.relative_to(ROOT))]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if episode==20000:
                training_evidence[str(seed)]={key:payload[key] for key in ('environment_steps','updates','decision_transitions','initial_parameter_hash','parameter_hash')}
    output.mkdir(parents=True)
    wrapper_hash=hashlib.sha256((ROOT/'src/ensembles.py').read_bytes()).hexdigest()
    source=output/'evaluation_source'
    for name in ('src/ensembles.py','src/policy_loading.py','src/environment.py','train_candidate.py','validate_final_learner.py'):
        destination=source/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes((ROOT/name).read_bytes())
    ensemble=LearnedQEnsemble(paths)
    ensemble_rows=evaluate(ensemble,200)
    write_csv(output/'ensemble_episodes.csv',ensemble_rows)
    summary=[{'policy':'uniform_all10_temporal_q','training_seed':'fixed_all5x2','scenarios':200,
              **{metric:float(np.mean([row[metric] for row in ensemble_rows])) for metric in METRICS}}]
    print('FIXED_ENSEMBLE '+json.dumps(summary[0]),flush=True)
    individual_rows=[]
    for seed in SEEDS:
        checkpoint=ROOT/f'candidates/final_v5_temporal_seed{seed}/seed{seed}_ep20000.pt'
        policy=load_snapshot_policy(checkpoint)
        rows=evaluate(policy,200)
        individual_rows.extend([{'training_seed':seed,**row} for row in rows])
        result={'policy':'individual20k_descriptive','training_seed':str(seed),'scenarios':200,
                **{metric:float(np.mean([row[metric] for row in rows])) for metric in METRICS}}
        summary.append(result);print('INDIVIDUAL '+json.dumps(result),flush=True)
    write_csv(output/'individual20k_episodes.csv',individual_rows)
    write_csv(output/'summary.csv',summary)
    assert hashlib.sha256((ROOT/'src/ensembles.py').read_bytes()).hexdigest()==wrapper_hash
    evidence={'status':'COMPLETE','purpose':'frozen deployment sanity and descriptive individual robustness; no member/subset/weight selection',
              'timestamp_utc':datetime.now(timezone.utc).isoformat(),'training_seeds':list(SEEDS),
              'temporal_checkpoints':list(TIMES),'uniform_members':10,'independent_training_runs':5,
              'independent_deployment_ensembles':1,'validation_scenario_rule':'600000+101*i',
              'validation_scenarios':200,'final_test_executed':False,'checkpoint_sha256':hashes,
              'configuration':family['configuration'],'source_sha256_common':family['source_sha256_common'],
              'ensemble_wrapper_sha256':wrapper_hash,'training_evidence':training_evidence,
              'ensemble':summary[0],
              'individual20k_return_mean':float(np.mean([row['return'] for row in summary[1:]])),
              'individual20k_return_seed_sd':float(np.std([row['return'] for row in summary[1:]],ddof=1))}
    (output/'manifest.json').write_text(json.dumps(evidence,indent=2)+'\n')
    family.update(status='TRAINING_COMPLETE_VALIDATED',completed_training_seeds=list(SEEDS),
                  checkpoint_sha256=hashes,training_evidence=training_evidence,
                  actual_total_environment_steps=sum(row['environment_steps'] for row in training_evidence.values()),
                  new_seed_validation_executed=True,new_seed_validation_path=str(output.relative_to(ROOT)),
                  new_seed_validation_ensemble_return=summary[0]['return'],
                  individual20k_return_mean=evidence['individual20k_return_mean'],
                  individual20k_return_seed_sd=evidence['individual20k_return_seed_sd'],final_test_executed=False)
    family_path.write_text(json.dumps(family,indent=2)+'\n')
    print('Complete frozen deployment sanity; no final scenarios accessed.',flush=True)

if __name__=='__main__':main()
