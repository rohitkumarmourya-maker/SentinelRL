#!/usr/bin/env python3
"""Freeze policy bytes/statistics before requesting parent authorization; no test run."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,json
import torch
from src.environment import EnvConfig
from src.policy_loading import load_snapshot_policy,checkpoint_members
from audit_scenario_seeds import audit
from src.comparators import comparator_seeds,make_comparator

ROOT=Path(__file__).resolve().parent
RUNTIME_FILES=('src/environment.py','src/belief.py','src/comparators.py','src/ensembles.py',
               'src/policy_loading.py','src/benchmark_stats.py','src/original_tabular.py','src/dqn.py','src/policies.py',
               'vendor/original_v1/dqn.py','vendor/original_v1/environment.py','vendor/original_v1/policies.py',
               'benchmark.py','benchmark_worker.py','audit_scenario_seeds.py','summarize_champion_benchmark.py',
               'provenance/original-file-hashes.json','archive/corrected_v2/provenance/delivery-file-hashes.json')


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--draft',type=Path,required=True)
 p.add_argument('--output',type=Path,default=ROOT/'protocol/frozen_comparison.json')
 args=p.parse_args()
 if args.output.exists():raise FileExistsError('frozen protocol already exists; preserve it')
 draft=json.loads(args.draft.read_text())
 if (ROOT/'src/environment.py').read_bytes()!=(ROOT/'archive/corrected_v2/src/environment.py').read_bytes():
  raise ValueError('environment changed from corrected assignment')
 names=[spec['name'] for spec in draft['policies']]
 primary_spec=next((spec for spec in draft['policies'] if spec['name']==draft['primary_policy']),None)
 if primary_spec is None or primary_spec['kind'] not in ('checkpoint','ensemble'):
  raise ValueError('primary must be a learned checkpoint family or learned-Q ensemble')
 primary_members=checkpoint_members(primary_spec)
 expected_seeds={17,29,43,71,89}
 if {m['training_seed'] for m in primary_members}!=expected_seeds:
  raise ValueError('primary must use exactly the five fresh training seeds')
 if primary_spec['kind']=='ensemble':
  if primary_spec.get('subtype')!='snapshot_learned' or len(primary_members)!=10:
   raise ValueError('primary must have ten uniform learned-Q temporal members')
  for seed in expected_seeds:
   if sorted(m['training_episodes'] for m in primary_members if m['training_seed']==seed)!=[10000,20000]:
    raise ValueError('each primary training seed requires genuine10k+20k snapshots')
 if len(names)!=len(set(names)) or draft['primary_policy'] not in names:raise ValueError('invalid policy names/primary')
 frozen={**draft,'status':'FROZEN_PRESPECIFIED','protocol_state':'FROZEN_PRESPECIFIED','frozen_utc':datetime.now(timezone.utc).isoformat(),
         'environment':asdict(EnvConfig()),'environment_sha256':hashlib.sha256((ROOT/'src/environment.py').read_bytes()).hexdigest(),
         'runtime_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in RUNTIME_FILES},
         'final_scenarios':10000,'final_seed_base':900000000,'final_seed_stride':103,'final_seed_formula':'900000000+103*i',
         'bootstrap_replicates':50000,'familywise_alpha':.05,
         'primary_metric':'undiscounted_episode_return','test_outcomes_used_for_selection':False,
         'ci_method':'paired common-scenario bootstrap; independent unequal trained-seed axes; fixed ensembles have one row; Bonferroni .05/K',
         'checkpoint_training_evidence':{},'include_fixed_checkpoint_comparisons':draft.get('include_fixed_checkpoint_comparisons',False)}
 for spec in frozen['policies']:
  kind=spec['kind']
  if kind in ('checkpoint','tabular_checkpoint') or kind=='ensemble' and spec['subtype']=='snapshot_learned':
   evidence={};hashes={}
   entries=checkpoint_members(spec)
   for member in entries:
    member_id=member['member_id'];relative=member['checkpoint'];seed=member['training_seed']
    path=(ROOT/relative).resolve()
    if kind=='tabular_checkpoint':
     from src.original_tabular import OriginalTabularPolicy
     policy=OriginalTabularPolicy(path);payload=json.loads(path.read_text());record=policy.metadata
     if record['seed']!=seed or record['environment_steps']!=48*record['episodes']:raise ValueError('tabular seed/time counters mismatch')
     if record['mode']!=spec.get('trained_mode'):raise ValueError('tabular training regime differs')
     expected_episodes=280 if record['mode']=='historical' else 20000
     if record['episodes']!=expected_episodes:raise ValueError('tabular budget label mismatch')
     hashes[member_id]=hashlib.sha256(path.read_bytes()).hexdigest();evidence[member_id]=record
     continue
    load_snapshot_policy(path)
    payload=torch.load(path,map_location='cpu',weights_only=True)
    actual_episodes=payload['environment_steps']//EnvConfig().horizon
    if seed!=payload['seed'] or member['training_episodes'] is not None and member['training_episodes']!=actual_episodes:
     raise ValueError('member seed/time differs from its actual checkpoint counters')
    if payload['parameter_hash']==payload['initial_parameter_hash']:raise ValueError('trained checkpoint did not change parameters')
    hashes[member_id]=hashlib.sha256(path.read_bytes()).hexdigest()
    evidence[member_id]={key:payload[key] for key in ('seed','config','environment_steps','updates','decision_transitions','initial_parameter_hash','parameter_hash')}
    evidence[member_id]['actual_training_episodes']=actual_episodes
   spec['checkpoint_sha256']=hashes
   spec['independent_policy_rows']=len(entries) if kind in ('checkpoint','tabular_checkpoint') else 1
   if kind=='ensemble':
    spec['fixed_trained_members']=len(entries);spec['unique_training_seeds']=sorted({m['training_seed'] for m in entries})
   frozen['checkpoint_training_evidence'][spec['name']]=evidence
  elif kind=='comparator':
   for seed in spec.get('model_seeds',comparator_seeds(spec['comparator'])):
    make_comparator(spec['comparator'],int(seed),600000)
   spec['independent_policy_rows']=len(spec.get('model_seeds',comparator_seeds(spec['comparator'])))
  elif kind=='ensemble':
   if spec['subtype'] not in ('original','previous'):raise ValueError('unknown ensemble subtype')
   spec['independent_policy_rows']=1;spec['fixed_trained_members']=len(spec.get('member_seeds',[17,29,43]))
  elif kind in ('threshold','no_op'):spec['independent_policy_rows']=1
  else:raise ValueError(kind)
 if 'required_comparators' not in frozen:
  frozen['required_comparators']=[spec['name'] for spec in frozen['policies'] if spec['name']!=frozen['primary_policy'] and not spec.get('report_only',False)]
 if not frozen['required_comparators'] or any(name not in names or name==frozen['primary_policy'] for name in frozen['required_comparators']):raise ValueError('invalid closed comparison family')
 extra_fixed=0
 if frozen['include_fixed_checkpoint_comparisons']:
  for spec in frozen['policies']:
   count=spec['independent_policy_rows']
   if spec['name'] in frozen['required_comparators'] and count>1:extra_fixed+=count
 frozen['bonferroni_comparators']=len(frozen['required_comparators'])+extra_fixed
 frozen['scenario_seed_audit']=audit()
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(frozen,indent=2)+'\n')
 digest=hashlib.sha256(args.output.read_bytes()).hexdigest()
 template={'authorized':False,'protocol_sha256':digest,'scenario_count':frozen['final_scenarios'],'scenario_base':900000000,'scenario_stride':103,
           'required_parent_review':'all policy weights/configuration/comparator definitions frozen before fresh test',
           'instructions':'Parent reviewer may write approved contents as provenance/FINAL_TEST_AUTHORIZATION.json. This template does not authorize evaluation.'}
 template_path=ROOT/'provenance/FINAL_TEST_AUTHORIZATION_TEMPLATE.json'
 template_path.write_text(json.dumps(template,indent=2)+'\n')
 print(json.dumps({'protocol':str(args.output),'sha256':digest,'policies':len(names),'required_comparators':frozen['bonferroni_comparators'],
                   'final_scenarios':frozen['final_scenarios'],'test_executed':False,'authorization':False},indent=2))


if __name__=='__main__':main()
