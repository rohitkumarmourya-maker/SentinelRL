"""Development validation of arithmetic learned-Q averaging, never final testing."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from src.champion import EnsembleChampionAgent
from train_candidate import evaluate, write_csv, METRICS


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--episodes',type=int,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[101,202]);p.add_argument('--scenarios',type=int,default=200)
    p.add_argument('--member-candidates',nargs='+',default=None)
    args=p.parse_args();root=Path(__file__).resolve().parent;directory=root/'candidates'/args.candidate
    member_candidates=args.member_candidates or [args.candidate]*len(args.seeds)
    if len(member_candidates)!=len(args.seeds): raise ValueError('one member candidate directory per training seed is required')
    paths=[root/'candidates'/candidate/f'seed{s}_ep{args.episodes}.pt' for candidate,s in zip(member_candidates,args.seeds)]
    destination=directory/f'validation_ensemble_ep{args.episodes}.csv'
    if destination.exists():raise FileExistsError(destination)
    policy=EnsembleChampionAgent.from_checkpoints(paths)
    rows=evaluate(policy,args.scenarios);write_csv(destination,rows)
    summary={'policy':'arithmetic_mean_learned_q','candidate':args.candidate,'training_episodes_per_member':args.episodes,
             'training_seeds':args.seeds,'member_candidates':member_candidates,'members':len(paths),'independent_ensembles':1,
             'validation_scenario_rule':'600000+101*i','validation_scenarios':args.scenarios,'test_scenarios_used':False,
             'checkpoint_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
             'ensemble_source_sha256':hashlib.sha256((root/'src/champion.py').read_bytes()).hexdigest(),
             **{metric:float(np.mean([row[metric] for row in rows])) for metric in METRICS}}
    (directory/f'validation_ensemble_ep{args.episodes}.json').write_text(json.dumps(summary,indent=2)+'\n')
    (directory/'ensemble_source.py').write_bytes((root/'src/champion.py').read_bytes())
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
