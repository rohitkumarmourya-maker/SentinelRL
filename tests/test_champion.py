import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.champion import ChampionAgent, ChampionConfig, PrioritizedReplay, EnsembleChampionAgent
from src.environment import SentinelEnv


class ChampionTests(unittest.TestCase):
    def test_argmax_is_learned_legal_q(self):
        agent=ChampionAgent(101,ChampionConfig(hidden_size=32))
        env=SentinelEnv(100); obs,info=env.reset(100); agent.reset(obs)
        with torch.no_grad():
            expected=agent.online(torch.from_numpy(agent.current_features)[None]).numpy()[0]
        self.assertEqual(agent.act(obs,info['action_mask']),int(np.argmax(np.where(info['action_mask'],expected,-np.inf))))
        # Filter inference is unchanged by arbitrary reward/metric quantities.
        b=agent.belief.probabilities.copy()
        env.episode_return=1000000; env.false_isolations=1000000
        np.testing.assert_array_equal(b,agent.belief.probabilities)

    def test_contextual_action_ranking_sees_other_hosts_defenses(self):
        config=ChampionConfig(hidden_size=32,architecture="contextual_host",representation="sufficient")
        agent=ChampionAgent(101,config)
        env=SentinelEnv(123);obs,_=env.reset(123);agent.reset(obs)
        tensor=torch.tensor(agent.current_features[None],requires_grad=True)
        q=agent.online(tensor)
        # Compare host0 and host1 isolation. Host4 isolation is public remote
        # context; a common state-value head cannot create this derivative.
        gradient=torch.autograd.grad(q[0,13]-q[0,14],tensor)[0]
        self.assertGreater(abs(float(gradient[0,4*9+7])),1e-9)
        # The earlier local-only shared action model has the expressivity gap.
        old=ChampionAgent(101,ChampionConfig(hidden_size=32,architecture="shared_host",representation="sufficient"))
        old.reset(obs);old_x=torch.tensor(old.current_features[None],requires_grad=True)
        old_q=old.online(old_x)
        old_gradient=torch.autograd.grad(old_q[0,13]-old_q[0,14],old_x)[0]
        self.assertAlmostEqual(float(old_gradient[0,4*9+7]),0.,places=9)

    def test_ensemble_is_mean_learned_q_with_one_causal_filter(self):
        members=[ChampionAgent(s,ChampionConfig(hidden_size=32)) for s in (101,202)]
        ensemble=EnsembleChampionAgent(members)
        env=SentinelEnv(123);obs,info=env.reset(123);ensemble.reset(obs)
        tensor=torch.from_numpy(members[0].current_features)[None]
        with torch.no_grad():
            q=torch.stack([member.online(tensor) for member in members]).mean(0).numpy()[0]
        expected=int(np.argmax(np.where(info['action_mask'],q,-np.inf)))
        self.assertEqual(ensemble.act(obs,info['action_mask']),expected)
        next_obs,_,_,_,next_info=env.step(expected)
        ensemble.act(next_obs,next_info['action_mask'])
        self.assertEqual(members[0].last_time,1)
        self.assertIsNone(members[1].last_time)

    def test_prioritized_replay_tree_sampling_and_importance(self):
        config=ChampionConfig(prioritized_replay=True,priority_alpha=1.)
        replay=PrioritizedReplay(7,101,config)
        for action in range(7):
            replay.add(np.zeros(126),action,0.,np.zeros(126),0.,np.ones(25,dtype=bool))
        replay.update_priorities(np.arange(7),np.arange(1.,8.)-.001)
        self.assertAlmostEqual(replay.tree[1],28.)
        batch=replay.sample(2800)
        observed=np.bincount(batch[7],minlength=7)/2800
        np.testing.assert_allclose(observed,np.arange(1.,8.)/28,atol=.001)
        self.assertTrue(torch.all(batch[6] > 0))
        self.assertTrue(torch.all(batch[6] <= 1))
        replay.update_priorities(np.array([3,3]),np.array([8.,9.]))
        self.assertAlmostEqual(replay.tree[replay.leaf_count+3],9.001)
        # Replacing circular replay leaves must preserve the root sum.
        for action in range(10):
            replay.add(np.zeros(126),action,0.,np.zeros(126),0.,np.ones(25,dtype=bool))
        self.assertAlmostEqual(replay.tree[1],replay.tree[replay.leaf_count:].sum())

    def test_gradient_checkpoint_and_exact_forced_tail(self):
        config=ChampionConfig(hidden_size=32,batch_size=4,learning_starts=4,n_step=1)
        agent=ChampionAgent(101,config)
        initial=agent.parameter_hash()
        env=SentinelEnv(901); obs,info=env.reset(901); agent.reset(obs)
        total=0.
        # Seeded legal exploration spends the tokens. Values are updated only
        # from actual collected rewards; there are no supplied target actions.
        for step in range(48):
            action=agent.act(obs,info['action_mask'],explore=True)
            ns,r,done,_,ni=env.step(action); total+=r
            agent.observe(obs,action,r,ns,done,ni['action_mask'])
            obs,info=ns,ni
        self.assertLess(agent.replay.size,48)
        self.assertAlmostEqual(float(agent.replay.r[:agent.replay.size].sum()),total,places=4)
        self.assertNotEqual(agent.parameter_hash(),initial)
        self.assertGreater(agent.updates,0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.pt'; agent.save(path,{'test':True}); loaded=ChampionAgent.load(path)
            self.assertEqual(loaded.parameter_hash(),agent.parameter_hash())
            obs,info=env.reset(777); agent.reset(obs); loaded.reset(obs)
            self.assertEqual(agent.act(obs,info['action_mask']),loaded.act(obs,info['action_mask']))


if __name__=='__main__': unittest.main()
