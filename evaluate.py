#!/usr/bin/env python3
"""Re-evaluate the exact bundled policies into a fresh output directory."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parent


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-model-run',action='store_true')
 p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
 p.add_argument('--full-frozen-test',action='store_true',help='Reproduce all10,000recorded final scenarios; default is20development scenarios')
 args=p.parse_args()
 command=[sys.executable,str(ROOT/'benchmark.py'),'--spec',str(ROOT/'protocol/frozen_comparison.json'),
          '--split','final' if args.full_frozen_test else 'validation','--workers',str(args.workers),'--output',str(args.output.resolve())]
 if not args.full_frozen_test:command+=['--episodes','20']
 print(' '.join(command),flush=True)
 if args.execute_model_run:subprocess.run(command,cwd=ROOT,check=True)
 else:print('Dry run. Add --execute-model-run for actual inference/evaluation.')


if __name__=='__main__':main()
