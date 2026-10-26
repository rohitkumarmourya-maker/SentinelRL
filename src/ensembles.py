"""Mean-Q deployment ensembles; member count is not independent ensemble count."""
from __future__ import annotations
from dataclasses import asdict
import inspect
import numpy as np
import torch
from .comparators import OriginalPolicy, PreviousPolicy, Lawful58Adapter
from .policy_loading import load_snapshot_policy


class LearnedQEnsemble:
    """Average only learned Q values for new pure graph/champion snapshot agents."""
    def __init__(self, checkpoints):
        self.members=[load_snapshot_policy(path) for path in checkpoints]
        if not self.members:
            raise ValueError('ensemble requires members')
        if any(getattr(member,'kind','pure')!='pure' for member in self.members):
            raise ValueError('learned-Q ensemble excludes prior/shield comparator agents')
        if len({member.config.gamma for member in self.members})!=1:
            raise ValueError('ensemble members must share objective discount')

        # A common verified feature mapping makes one causal filter sufficient.
        first=self.members[0]
        for other in self.members[1:]:
            if asdict(other.config)!=asdict(first.config) or asdict(other.env_config)!=asdict(first.env_config):
                raise ValueError('shared-feature ensemble requires identical feature/configuration mappings')
            for method in ('reset','_prepare','_represent'):
                left=getattr(type(first),method,None);right=getattr(type(other),method,None)
                if (left is None)!=(right is None) or left is not None and inspect.getsource(left)!=inspect.getsource(right):
                    raise ValueError('member feature-runtime methods differ')
        self.feature_agent=first

    def reset(self,observation,mask=None):
        self.feature_agent.reset(observation,mask)

    def act(self,observation,mask,explore=False):
        if explore:
            raise ValueError('ensemble is a frozen deployment policy, not a training teacher')
        features=self.feature_agent._prepare(observation)
        legal=np.flatnonzero(mask)
        if len(legal)==1:
            action=int(legal[0])
        else:
            with torch.no_grad():
                scores=np.mean([member.online(torch.from_numpy(features)[None]).cpu().numpy()[0]
                                for member in self.members],axis=0)
            action=int(np.argmax(np.where(mask,scores,-np.inf)))
        self.feature_agent.last_action=action
        self.feature_agent.acted_features=features.copy()
        self.feature_agent.acted_mask=np.array(mask,copy=True)
        return action


class OriginalQEnsemble:
    """Original checkpoint mean-Q transfer, retaining the original prior/shield."""
    def __init__(self,seeds=(17,29,43),no_stealth=False):
        kind='original_no_stealth_transfer' if no_stealth else 'original_full_transfer'
        policies=[OriginalPolicy(kind,seed) for seed in seeds]
        self.members=[policy.agent for policy in policies]
        self.adapter=Lawful58Adapter()

    def reset(self,observation,mask=None):self.adapter.reset(observation)

    def act(self,observation,mask,explore=False):
        features=self.adapter.advance(observation)
        mask=self.adapter.legacy_mask(observation,mask)
        with torch.no_grad():
            scores=np.mean([member.online(torch.from_numpy(features)[None]).cpu().numpy()[0]
                            for member in self.members],axis=0)
        nodes=features[:54].reshape(6,9)
        for host in range(6):
            if mask[13+host] and not(nodes[host,2]>.5 and nodes[host,3]>.5) and nodes[host,0]<.70:
                scores[13+host]-=10.
        action=int(np.argmax(np.where(mask,scores,-np.inf)))
        self.adapter.previous_action=action
        return action


class PreviousQEnsemble:
    """Three old corrected learned-Q models, with their original input transform."""
    def __init__(self,kind='previous_ddqn',seeds=(17,29,43)):
        if kind=='previous_tabular':raise ValueError('not a neural Q ensemble')
        self.members=[PreviousPolicy(kind,seed).agent for seed in seeds]

    def reset(self,observation,mask=None):pass

    def act(self,observation,mask,explore=False):
        from .dqn import transform_observation
        with torch.no_grad():
            scores=np.mean([member.online(torch.from_numpy(transform_observation(observation,member.observation_mode))[None]).cpu().numpy()[0]
                            for member in self.members],axis=0)
        return int(np.argmax(np.where(mask,scores,-np.inf)))
