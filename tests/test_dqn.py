"""Model correctness tests: real gradient, target, legal-action and checkpoint checks."""
import pytest
pytestmark = pytest.mark.model


def test_double_dqn_online_selection_target_evaluation_and_terminal_mask():
    import torch
    from src.dqn import double_dqn_targets
    rewards = torch.tensor([1., 2.])
    dones = torch.tensor([0., 1.])
    online = torch.tensor([[100., 2., 3.], [9., 8., 7.]])
    target = torch.tensor([[400., 50., 6.], [1., 2., 3.]])
    masks = torch.tensor([[False, True, True], [True, True, True]])
    actual = double_dqn_targets(rewards, dones, online, target, masks, .9)
    torch.testing.assert_close(actual, torch.tensor([6.4, 2.]))


def test_actual_gradient_updates_change_parameters_and_sync_target():
    import numpy as np
    import torch
    from src.dqn import DQNAgent, DQNConfig
    config = DQNConfig(batch_size=2, replay_capacity=8, learning_starts=2, train_every=1, target_sync_updates=1)
    agent = DQNAgent(seed=7, config=config)
    initial = agent.parameter_hash()
    state, next_state, mask = np.zeros(56, dtype=np.float32), np.ones(56, dtype=np.float32), np.ones(25, dtype=bool)
    agent.observe(state, 1, 5., next_state, False, mask)
    update = agent.observe(next_state, 2, -3., state, True, mask)
    assert update['gradient_norm'] > 0 and agent.updates == 1
    assert initial != agent.parameter_hash()
    for a, b in zip(agent.online.parameters(), agent.target.parameters()):
        torch.testing.assert_close(a, b)
    assert all(p.grad is None for p in agent.target.parameters())


def test_checkpoint_roundtrip_and_legal_action(tmp_path):
    import numpy as np
    from src.dqn import DQNAgent
    agent = DQNAgent(seed=3)
    state, mask = np.zeros(56, dtype=np.float32), np.zeros(25, dtype=bool)
    mask[7] = True
    assert agent.act(state, mask) == 7
    path = tmp_path / 'model.pt'
    agent.save(path, {'version': 'synthetic unit test only'})
    loaded = DQNAgent.load(path)
    assert loaded.parameter_hash() == agent.parameter_hash()
    assert loaded.act(state, mask) == 7


def test_no_memory_ablation_only_removes_documented_features():
    import numpy as np
    from src.dqn import transform_observation
    state = np.ones(56, dtype=np.float32)
    result = transform_observation(state, 'no_memory')
    assert (result[:54].reshape(6, 9)[:, 1:4] == 0).all()
    assert (result[-2:] == 1).all()
    assert (state == 1).all()
