#!/usr/bin/env python3
"""Reproduce fresh champion training from its exact immutable class/configuration."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parent


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-model-run',action='store_true')
 p.add_argument('--name',default='reproduction_champion');p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43,71,89])
 args=p.parse_args();primary=json.loads((ROOT/'models/deployment.json').read_text())['policy']
 template=next(member['checkpoint'] for member in primary['members'] if member['training_seed']==17 and member['training_episodes']==20000)
 command=[sys.executable,str(ROOT/'train_final.py'),'--template',str(ROOT/template),'--name',args.name,'--seeds',*map(str,args.seeds)]
 print(' '.join(command),flush=True)
 if args.execute_model_run:subprocess.run(command,cwd=ROOT,check=True)
 else:print('Dry run. Add --execute-model-run for real 20k-episode training for every requested seed.')


if __name__=='__main__':main()
