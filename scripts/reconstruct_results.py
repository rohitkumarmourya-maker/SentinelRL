#!/usr/bin/env python3
"""Rebuild consolidated raw CSV from tracked gzip/chunks after restoring Git history."""
from pathlib import Path
import csv,gzip,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'results/final_benchmark';target=run/'episodes.csv'
if target.exists():print('Consolidated raw CSV already exists; preserved.')
elif target.with_suffix('.csv.gz').exists():
 with gzip.open(target.with_suffix('.csv.gz'),'rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
 print('Restored episodes.csv from tracked gzip.')
else:
 manifest=json.loads((run/'manifest.json').read_text());spec=manifest['frozen_specification'];writer=None
 with target.open('x',newline='') as output:
  for index,policy in enumerate(spec['policies']):
   seeds=list(map(int,policy['checkpoints'])) if policy['kind'] in ('checkpoint','tabular_checkpoint') else policy.get('model_seeds',[0])
   for seed in seeds:
    paths=sorted((run/'chunks').glob(f'p{index:02d}_seed{seed}_*.csv'))
    for path in paths:
     with path.open() as f:
      for row in csv.DictReader(f):
       for key,value in list(row.items()):
        if key in ('policy','action_sequence','reward_sequence'):continue
        row[key]=float(value) if key not in ('model_seed','environment_seed') and not key.startswith('action_') else int(value)
       if writer is None:writer=csv.DictWriter(output,fieldnames=list(row));writer.writeheader()
       writer.writerow(row)
 print('Reconstructed episodes.csv from immutable chunks in frozen protocol order.')
index_path=run/'RAW_DATA_INDEX.json'
if index_path.exists():
 h=hashlib.sha256()
 with target.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 assert h.hexdigest()==json.loads(index_path.read_text())['episodes_csv_sha256']
 print('Reconstructed raw-data SHA-256 matches original.')
