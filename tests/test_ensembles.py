from pathlib import Path
import numpy as np
from src.ensembles import OriginalQEnsemble
from src.comparators import OriginalPolicy
from src.environment import SentinelEnv


def test_single_original_member_ensemble_matches_original_policy_actions():
    policies=[OriginalPolicy('original_full_transfer',17),OriginalQEnsemble((17,))]
    envs=[SentinelEnv(seed=600000),SentinelEnv(seed=600000)]
    pairs=[env.reset(600000) for env in envs]
    for policy,(obs,info) in zip(policies,pairs):policy.reset(obs,info['action_mask'])
    for _ in range(48):
        actions=[policy.act(obs,info['action_mask']) for policy,(obs,info) in zip(policies,pairs)]
        assert actions[0]==actions[1]
        pairs=[(result[0],result[4]) for result in [env.step(action) for env,action in zip(envs,actions)]]
    assert envs[0].metrics()==envs[1].metrics()
