import numpy as np
import torch
from src.dqn import DQNAgent, DQNConfig
from src.environment import SentinelEnv, OBS_DIM, N_ACTIONS


def test_masking_and_checkpoint_round_trip(tmp_path):
    torch.set_num_threads(1)
    cfg = DQNConfig(batch_size=4, learning_starts=4, replay_capacity=20, hidden_size=32, train_every=1)
    agent = DQNAgent(OBS_DIM, N_ACTIONS, seed=3, config=cfg)
    env = SentinelEnv(seed=3)
    obs, info = env.reset(seed=3)
    mask = info["action_mask"]
    action = agent.act(obs, mask)
    assert mask[action]
    path = tmp_path / "model.pt"
    agent.save(str(path), metadata={"test": True})
    restored = DQNAgent.load(str(path))
    action2 = restored.act(obs, mask)
    assert mask[action2]
    with torch.no_grad():
        q1 = agent.online(torch.as_tensor(obs).unsqueeze(0))
        q2 = restored.online(torch.as_tensor(obs).unsqueeze(0))
    torch.testing.assert_close(q1, q2)


def test_training_update_returns_finite_loss():
    torch.set_num_threads(1)
    cfg = DQNConfig(batch_size=4, learning_starts=4, replay_capacity=20, hidden_size=32, train_every=1)
    agent = DQNAgent(OBS_DIM, N_ACTIONS, seed=4, config=cfg)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    next_obs = np.ones(OBS_DIM, dtype=np.float32) * 0.1
    mask = np.ones(N_ACTIONS, dtype=bool)
    for i in range(8):
        agent.steps += 1
        agent.remember(obs, i % N_ACTIONS, 0.5, next_obs, False, mask)
    loss = agent.learn()
    assert loss is not None and np.isfinite(loss)
