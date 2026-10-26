"""Original six-field Q-learning algorithm, historical transfer or lawful retrain."""
from __future__ import annotations
from pathlib import Path
import hashlib,json
import numpy as np
from .comparators import Lawful58Adapter,verify_original_source,ROOT
from vendor.original_v1.policies import TabularQLearningAgent


def save_original_tabular(agent,path,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rows=[{'state':list(key),'q':value.tolist()} for key,value in sorted(agent.q_table.items())]
    payload={'format_version':5,'algorithm':'original_tabular_q','alpha':agent.alpha,'gamma':agent.gamma,
             'n_actions':25,'encoding_fields':6,'table':rows,'metadata':metadata}
    path.write_text(json.dumps(payload,separators=(',',':'))+'\n')


class OriginalTabularPolicy:
    def __init__(self,path):
        verify_original_source();self.path=Path(path)
        payload=json.loads(self.path.read_text())
        if payload.get('format_version')!=5 or payload.get('algorithm')!='original_tabular_q' or payload.get('n_actions')!=25 or payload.get('encoding_fields')!=6:
            raise ValueError('incompatible original tabular checkpoint')
        if payload['alpha']!=.18 or payload['gamma']!=.97:raise ValueError('original tabular algorithm changed')
        self.metadata=payload['metadata']
        for name,expected in self.metadata['source_sha256'].items():
            source=self.path.parent/'source'/name
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
                raise ValueError('tabular immutable source mismatch:'+name)
        self.agent=TabularQLearningAgent(seed=self.metadata['seed'],alpha=payload['alpha'],gamma=payload['gamma'])
        for row in payload['table']:
            if len(row['state'])!=6 or len(row['q'])!=25 or not np.isfinite(row['q']).all():raise ValueError('invalid Q-table')
            self.agent.q_table[tuple(row['state'])]=np.array(row['q'],dtype=np.float32)
        self.adapter=Lawful58Adapter()

    def reset(self,observation,mask=None):self.adapter.reset(observation)

    def act(self,observation,mask,explore=False):
        features=self.adapter.advance(observation)
        legal=self.adapter.legacy_mask(observation,mask)
        action=self.agent.act(features,legal,epsilon=0.)
        self.adapter.previous_action=action
        return int(action)
