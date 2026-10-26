#!/usr/bin/env python3
"""Copy selected model/source evidence into primary portable paths before freeze."""
from pathlib import Path
import hashlib,json,shutil

ROOT=Path(__file__).resolve().parent
source=ROOT/'protocol/comparison_draft.json'
spec=json.loads(source.read_text());copied={}
for policy in spec['policies']:
 entries=policy.get('members',[])+[{'checkpoint':path} for path in policy.get('checkpoints',{}).values()]
 mapping={}
 for member in entries:
  relative=member['checkpoint'];original=ROOT/relative
  if not original.is_file():raise FileNotFoundError(original)
  parent=original.parent
  group='_'.join(parent.relative_to(ROOT).parts[1:])
  destination=ROOT/'models/runs'/group;destination.mkdir(parents=True,exist_ok=True)
  target=destination/original.name
  if target.exists():
   if target.read_bytes()!=original.read_bytes():raise ValueError('primary model collision')
  else:shutil.copy2(original,target)
  if not (destination/'source').exists():shutil.copytree(parent/'source',destination/'source')
  else:
   for path in (parent/'source').rglob('*'):
    if path.is_file() and '__pycache__' not in path.parts:
     if path.read_bytes()!=(destination/'source'/path.relative_to(parent/'source')).read_bytes():raise ValueError('source snapshot collision')
  normalized=str(target.relative_to(ROOT));mapping[relative]=normalized;copied[normalized]=hashlib.sha256(target.read_bytes()).hexdigest()
 for member in policy.get('members',[]):member['checkpoint']=mapping[member['checkpoint']]
 if 'checkpoints' in policy:policy['checkpoints']={seed:mapping[path] for seed,path in policy['checkpoints'].items()}
ROOT.joinpath('protocol/prepared_comparison_draft.json').write_text(json.dumps(spec,indent=2)+'\n')
ROOT.joinpath('models/selected-model-hashes.json').write_text(json.dumps(copied,indent=2)+'\n')
results=ROOT/'results';results.mkdir(exist_ok=True)
for seed in (17,29,43,71,89):
 source_csv=ROOT/f'candidates/final_v5_temporal_seed{seed}/training_seed{seed}.csv'
 shutil.copy2(source_csv,results/f'training_champion_seed{seed}.csv')
shutil.copy2(ROOT/'candidates/final_v5_temporal_family.json',results/'CHAMPION_TRAINING_FAMILY.json')
shutil.copy2(ROOT/'validation/final_v5_temporal_sanity/summary.csv',results/'development_sanity_summary.csv')
print('Prepared',len(copied),'distinct portable model files; primary five training CSVs copied; no evaluation.')
