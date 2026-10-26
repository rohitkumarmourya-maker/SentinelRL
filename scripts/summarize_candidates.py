"""Collect development results without touching final evaluation scenarios."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
rows=[]; candidates=[]
for directory in sorted((ROOT/'candidates').iterdir()):
    manifest=directory/'manifest.json'
    if not manifest.exists(): continue
    payload=json.loads(manifest.read_text())
    summary=directory/'validation_summary.csv'
    if summary.exists():
        with summary.open() as f:
            current=list(csv.DictReader(f))
        for row in current:
            row['candidate']=directory.name
            rows.append(row)
        configurations={int(r['episodes']) for r in current if r['policy']=='champion'}
        means=[]
        for episodes in sorted(configurations):
            selected=[r for r in current if r['policy']=='champion' and int(r['episodes'])==episodes]
            means.append({'episodes':episodes,'seeds':sorted(int(r['seed']) for r in selected),
                          **{key:float(np.mean([float(r[key]) for r in selected])) for key in current[0] if key not in ('policy','seed','episodes','candidate')}})
        payload['validation_means']=means
    candidates.append(payload)
(ROOT/'candidates'/'DEVELOPMENT_INDEX.json').write_text(json.dumps(candidates,indent=2)+'\n')
if rows:
    with (ROOT/'candidates'/'DEVELOPMENT_SUMMARY.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
for candidate in candidates:
    for result in candidate.get('validation_means',[]):
        print(candidate['name'],result['episodes'],result['seeds'],round(result['return'],6))
