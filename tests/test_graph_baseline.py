import numpy as np
from src.environment import SentinelEnv
from src.graph_baseline import GraphBaselineAgent, GraphConfig


def test_graph_pure_gradient_and_checkpoint_roundtrip(tmp_path):
    config=GraphConfig(batch_size=2,replay_capacity=100,learning_starts=2)
    agent=GraphBaselineAgent(7,config,kind='pure')
    initial=agent.parameter_hash()
    env=SentinelEnv(seed=3);obs,info=env.reset(3);agent.reset(obs)
    for _ in range(48):
        action=agent.act(obs,info['action_mask'],explore=True)
        nxt,r,done,_,ni=env.step(action)
        agent.observe(obs,action,r,nxt,done,ni['action_mask']);obs,info=nxt,ni
    assert agent.updates>0 and agent.parameter_hash()!=initial
    assert not agent.online.use_topological_prior
    path=tmp_path/'graph.pt';agent.save(path)
    loaded=GraphBaselineAgent.load(path)
    assert loaded.kind=='pure' and loaded.parameter_hash()==agent.parameter_hash()
    obs,info=env.reset(9);loaded.reset(obs)
    assert info['action_mask'][loaded.act(obs,info['action_mask'])]
