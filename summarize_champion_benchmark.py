#!/usr/bin/env python3
"""Summarize a completed common-environment run; no policy/model execution."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from src.benchmark_stats import bootstrap_families


def write_csv(path,rows):
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
 p.add_argument('--primary');p.add_argument('--bootstrap',type=int)
 p.add_argument('--include-fixed-checkpoints',action='store_true')
 args=p.parse_args();rows=list(csv.DictReader((args.run/'episodes.csv').open()))
 policies=sorted({r['policy'] for r in rows});scenarios=sorted({int(r['environment_seed']) for r in rows})
 seeds={policy:sorted({int(r['model_seed']) for r in rows if r['policy']==policy}) for policy in policies}
 index={(r['policy'],int(r['model_seed']),int(r['environment_seed'])):r for r in rows}
 if len(index)!=len(rows):raise ValueError('duplicate raw comparison row')
 metrics=['return','total_compromises','critical_compromises','uncontained_at_end','false_isolations','successful_isolations','action_costs','invalid_actions','confirmed_fraction','censored_confirmation_delay']
 manifest=json.loads((args.run/'manifest.json').read_text())
 specification=manifest.get('frozen_specification') or {}
 args.primary=args.primary or specification.get('primary_policy','champion')
 args.bootstrap=args.bootstrap or specification.get('bootstrap_replicates',50000)
 if manifest.get('split')=='final' and (args.primary!=specification['primary_policy'] or args.bootstrap!=specification['bootstrap_replicates']):
  raise ValueError('primary/bootstrap must match frozen final protocol')
 comparators=specification.get('required_comparators')
 if manifest.get('split')=='final' and args.include_fixed_checkpoints!=specification.get('include_fixed_checkpoint_comparisons',False):
  raise ValueError('fixed-checkpoint comparison choice must match the prespecified protocol')
 summary=[];paired=[];per_seed=[]
 for metric in metrics:
  arrays={policy:np.array([[float(index[policy,seed,scenario][metric]) for scenario in scenarios] for seed in seeds[policy]]) for policy in policies}
  metric_comparators=list(comparators) if comparators is not None else [name for name in arrays if name!=args.primary]
  if args.include_fixed_checkpoints:
   for policy in policies:
    if policy in metric_comparators and len(seeds[policy])>1:
     for i,seed in enumerate(seeds[policy]):
      name=policy+'__fixed_seed'+str(seed)
      arrays[name]=arrays[policy][i:i+1];metric_comparators.append(name)
  stats,differences=bootstrap_families(arrays,args.primary,repetitions=args.bootstrap if metric=='return' else 2000,comparators=metric_comparators)
  summary.extend({'metric':metric,**r} for r in stats)
  paired.extend({'metric':metric,**r} for r in differences)
  for policy in policies:
   for seed in seeds[policy]:
    values=[float(index[policy,seed,scenario][metric]) for scenario in scenarios]
    per_seed.append({'policy':policy,'model_seed':seed,'metric':metric,'mean':float(np.mean(values)),'scenario_sd':float(np.std(values,ddof=1))})
 write_csv(args.run/'summary_intervals.csv',summary)
 write_csv(args.run/'paired_comparisons.csv',paired)
 write_csv(args.run/'per_seed_metrics.csv',per_seed)
 primary=[r for r in paired if r['metric']=='return']
 (args.run/'inference_status.json').write_text(json.dumps({'primary':args.primary,'bootstrap_replicates':args.bootstrap,
    'method':'crossed bootstrap; common scenario draws, independent unequal model-seed axes; Bonferroni alpha.05/K',
    'comparators':len(primary),'all_primary_return_familywise_lower_positive':all(r['familywise_lower_positive'] for r in primary),
    'all_primary_return_point_differences_positive':all(r['mean_difference']>0 for r in primary),
    'secondary_intervals_exploratory':True,'include_fixed_checkpoint_comparisons':args.include_fixed_checkpoints},indent=2)+'\n')
 print(json.dumps(primary,indent=2))


if __name__=='__main__':main()
