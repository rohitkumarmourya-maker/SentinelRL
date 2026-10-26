#!/usr/bin/env python3
"""Prespecified modest lawful-threshold validation grid; never uses final tests."""
from pathlib import Path
from dataclasses import asdict
import csv,json,time
import numpy as np
from src.comparators import BeliefThresholdPolicy
from src.environment import EnvConfig
from benchmark import run_episode

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'validation/threshold_grid'
OUTPUT.mkdir(parents=True,exist_ok=False)
configs=[{'isolation_threshold':iso,'inspection_threshold':inspect,'protection':protect}
         for iso in (.5,.65,.8,.9,.98) for inspect in (1.1,.2,.4,.6) for protect in (False,True)]
(OUTPUT/'grid.json').write_text(json.dumps({'scenarios':200,'seed_formula':'600000+101*i',
              'primary_selection':'maximum mean episodic return; ties prefer first prespecified config',
              'environment':asdict(EnvConfig()),'configurations':configs,'final_test_used':False},indent=2)+'\n')
rows=[];summary=[];start=time.monotonic()
for index,config in enumerate(configs):
 policy=BeliefThresholdPolicy(**config);current=[]
 for i in range(200):
  row={'configuration':index,'scenario_index':i,'environment_seed':600000+101*i,**run_episode(policy,600000+101*i,EnvConfig())}
  rows.append(row);current.append(row)
 summary.append({'configuration':index,**config,**{metric:float(np.mean([r[metric] for r in current])) for metric in ('return','critical_compromises','false_isolations','confirmed_fraction')}})
 print(index,config,summary[-1]['return'],flush=True)
with (OUTPUT/'episodes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (OUTPUT/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
best=max(summary,key=lambda row:row['return'])
(OUTPUT/'selected.json').write_text(json.dumps({'selection':best,'elapsed_seconds':time.monotonic()-start,'final_test_used':False},indent=2)+'\n')
print('BEST',json.dumps(best),flush=True)
