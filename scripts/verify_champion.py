#!/usr/bin/env python3
"""Verify sealed runtime/models and completed evidence; optional exact saved-episode replay."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark import load_spec_policy,run_episode
from src.environment import EnvConfig

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',action='store_true');p.add_argument('--deep',action='store_true')
 args=p.parse_args();protocol_path=ROOT/'protocol/frozen_comparison.json';spec=json.loads(protocol_path.read_text())
 authorization=json.loads((ROOT/'provenance/FINAL_TEST_AUTHORIZATION.json').read_text())
 assert authorization['authorized'] and digest(protocol_path)==authorization['protocol_sha256']
 for name,expected in spec['runtime_hashes'].items():assert digest(ROOT/name)==expected,name
 checked=set()
 for policy in spec['policies']:
  pairs=[(m['member_id'],m['checkpoint']) for m in policy.get('members',[])]+list(policy.get('checkpoints',{}).items())
  for key,path in pairs:
   assert digest(ROOT/path)==policy['checkpoint_sha256'][key],path;checked.add(path)
 run=ROOT/'results/final_benchmark';manifest=json.loads((run/'manifest.json').read_text())
 assert manifest['status']=='COMPLETE' and manifest['coverage_unique_complete']
 count=0
 for record in manifest['worker_chunks']:
  matches=list((run/'chunks').glob('*.json')) if False else None
  filename=f"p{next(i for i,p in enumerate(spec['policies']) if p['name']==record['policy']):02d}_seed{record['model_seeds'][0]}_{record['scenario_first_index']:05d}.csv"
  assert digest(run/'chunks'/filename)==record['sha256'],filename
  count+=record['rows']
 assert count==manifest['rows']
 raw=run/'episodes.csv'
 if not raw.exists():
  import subprocess
  subprocess.run([sys.executable,str(ROOT/'scripts/reconstruct_results.py')],check=True,cwd=ROOT)
 if args.deep:
  seen=set()
  with raw.open() as f:
   for row in csv.DictReader(f):
    key=(row['policy'],row['model_seed'],row['environment_seed']);assert key not in seen;seen.add(key)
    scenario=int(row['environment_seed']);assert scenario>=900000000 and (scenario-900000000)%103==0 and (scenario-900000000)//103<10000
    actions=json.loads(row['action_sequence']);rewards=json.loads(row['reward_sequence'])
    assert len(actions)==len(rewards)==48 and sum(int(row[f'action_{a}']) for a in range(25))==48
    assert np.isclose(sum(rewards),float(row['return']),rtol=0,atol=1e-8)
    assert int(float(row['invalid_actions']))==0 and float(row['action_costs'])<=6
  assert len(seen)==count
 if args.replay:
  with raw.open() as f:expected=next(csv.DictReader(f))
  primary=next(p for p in spec['policies'] if p['name']==spec['primary_policy'])
  policy=load_spec_policy(primary,0,int(expected['environment_seed']),EnvConfig())
  actual=run_episode(policy,int(expected['environment_seed']),EnvConfig())
  for key,value in actual.items():
   if key in ('action_sequence','reward_sequence'):assert json.loads(value)==json.loads(expected[key]),key
   else:assert np.isclose(value,float(expected[key]),rtol=0,atol=1e-9),key
 print(json.dumps({'status':'PASSED','sealed_protocol_sha256':digest(protocol_path),'selected_model_files':len(checked),
                   'raw_episode_rows':count,'runtime_hashes_verified':len(spec['runtime_hashes']),
                   'exact_saved_episode_replay':args.replay,'deep_trace_checks':args.deep},indent=2))


if __name__=='__main__':main()
