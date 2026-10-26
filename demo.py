#!/usr/bin/env python3
"""Run the bundled frozen10-network learned-Q deployment on a public demo scenario."""
from pathlib import Path
import argparse,json
from benchmark import load_spec_policy
from src.environment import SentinelEnv,EnvConfig,action_name

ROOT=Path(__file__).resolve().parent


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=7)
 p.add_argument('--steps',type=int,default=8);p.add_argument('--deployment',type=Path,default=ROOT/'models/deployment.json')
 args=p.parse_args()
 if args.steps<1: p.error('steps must be positive')
 if 900000000<=args.seed<=901029897 and not (ROOT/'results/final_benchmark/manifest.json').exists():p.error('reserved scenarios are not demonstration data')
 spec=json.loads(args.deployment.read_text())['policy'];policy=load_spec_policy(spec,0,args.seed,EnvConfig())
 env=SentinelEnv(args.seed);obs,info=env.reset(args.seed);policy.reset(obs,info['action_mask'])
 print('SentinelRL champion: uniform learned-Q mean of 10 snapshots from 5 independent training seeds.')
 print('Demo scenario only; benchmark conclusions come from the frozen comparison CSV.')
 for _ in range(min(args.steps,48)):
  action=policy.act(obs,info['action_mask'],explore=False)
  obs,reward,done,_,info=env.step(action)
  print(f't={env.t:02d} action={action_name(action)} budget={env.budget} alerts={env.alert.astype(int).tolist()} reward={reward:.4f}')
  if done:break
 if env.t==48:print(json.dumps(env.metrics(),indent=2))


if __name__=='__main__':main()
