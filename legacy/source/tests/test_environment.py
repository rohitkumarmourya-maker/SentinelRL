import numpy as np
from src.environment import SentinelEnv, N_ACTIONS, OBS_DIM, WAIT


def test_reset_and_step_shapes_and_finite_rewards():
    env = SentinelEnv(seed=11)
    obs, info = env.reset(seed=11)
    assert obs.shape == (OBS_DIM,)
    assert np.isfinite(obs).all()
    assert np.all((obs >= 0.0) & (obs <= 1.0))
    assert info["action_mask"].shape == (N_ACTIONS,)
    assert info["action_mask"][WAIT]
    for _ in range(env.config.horizon):
        mask = info["action_mask"]
        action = int(np.flatnonzero(mask)[0])
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (OBS_DIM,)
        assert np.isfinite(obs).all()
        assert np.all((obs >= 0.0) & (obs <= 1.0))
        assert np.isfinite(reward)
        assert info["action_mask"][WAIT]
        if terminated:
            assert not truncated
            assert "episode_summary" in info
            break
    assert terminated


def test_seeded_transition_reproducibility():
    def trajectory():
        env = SentinelEnv(seed=32)
        obs, info = env.reset(seed=32)
        out = []
        for _ in range(10):
            valid = np.flatnonzero(info["action_mask"])
            action = int(valid[-1])
            obs, reward, terminated, truncated, info = env.step(action)
            out.append((obs.copy(), reward, terminated))
            if terminated: break
        return out
    a, b = trajectory(), trajectory()
    assert len(a) == len(b)
    for (oa, ra, da), (ob, rb, db) in zip(a, b):
        np.testing.assert_allclose(oa, ob)
        assert ra == rb and da == db


def test_invalid_action_penalized_and_wait_always_valid():
    env = SentinelEnv(seed=5)
    env.reset(seed=5)
    # Force an unavailable action by spending the budget then attempting a patch.
    env.budget = 0
    _, r, _, _, info = env.step(2)
    assert r < 0
    assert info["invalid_action"]
    assert info["action_mask"][WAIT]


def test_budget_never_negative_and_terminates():
    env = SentinelEnv(seed=7)
    obs, info = env.reset(seed=7)
    for _ in range(env.config.horizon):
        valid = np.flatnonzero(info["action_mask"])
        action = int(valid[-1])
        obs, r, done, truncated, info = env.step(action)
        assert env.budget >= 0
        if done: break
    assert done
    assert env.t == env.config.horizon
