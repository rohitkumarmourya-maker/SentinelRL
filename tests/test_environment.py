"""Executed tests cover the simulator only: no policy/model inference or training."""
from dataclasses import replace
import numpy as np
import pytest
from src.environment import (SentinelEnv, EnvConfig, OBS_DIM, N_ACTIONS, FEATURES_PER_HOST,
                             N_HOSTS, CRITICALITY, VULNERABILITY, WAIT, INSPECT_BASE,
                             PATCH_BASE, ISOLATE_BASE, DECOY_BASE)


def controlled_env(config=None):
    env = SentinelEnv(seed=1, config=config)
    env.compromised[:] = False
    env.compromised[0] = True
    env.first_compromise_time[:] = -1
    env.first_compromise_time[0] = 0
    env.spread_uniforms[:] = 1.
    env.sensor_uniforms[:] = 1.
    env.alert[:] = False
    env.alert_ema[:] = 0
    return env


def test_observation_contract():
    env = SentinelEnv(seed=17)
    observation, info = env.reset(17)
    assert observation.shape == (OBS_DIM,) == (56,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert ((0 <= observation) & (observation <= 1)).all()
    assert info['action_mask'].shape == (N_ACTIONS,) == (25,)
    assert info['action_mask'].all()


def test_latent_state_and_accounting_do_not_leak_into_observation_or_mask():
    env = SentinelEnv(seed=4)
    before, mask = env._observation().copy(), env.action_mask().copy()
    env.compromised[:] = ~env.compromised
    env.first_confirmation_time[:] = 0
    env.first_compromise_time[:] = 0
    env.false_isolations = 5
    env.successful_isolations = 3
    env.episode_return = -999
    np.testing.assert_array_equal(before, env._observation())
    np.testing.assert_array_equal(mask, env.action_mask())


def test_reset_does_not_encode_actual_entry_when_sensors_uninformative():
    config = EnvConfig(alert_sensitivity=0., false_positive_rate=0.)
    envs = [SentinelEnv(seed=i, config=config) for i in range(20)]
    assert len({int(np.argmax(env.compromised)) for env in envs}) == 2
    for env in envs[1:]:
        np.testing.assert_array_equal(envs[0]._observation(), env._observation())


def test_hidden_infection_preserves_stale_clean_inspection():
    env = controlled_env()
    env.spread_uniforms[0, 0, 1] = 0
    observation, _, _, _, _ = env.step(INSPECT_BASE + 1)
    assert env.compromised[1]
    assert env.inspection_result[1] == 0
    assert env.inspection_time[1] == 0
    node = observation.reshape(-1)[:54].reshape(N_HOSTS, FEATURES_PER_HOST)[1]
    assert node[2] == .5  # old clean snapshot, even though hidden infection happened
    assert env.action_mask()[INSPECT_BASE + 1]


def test_no_hidden_resets_of_inspection_history_during_wait():
    env = controlled_env()
    env.inspection_result[1], env.inspection_time[1] = 0, 0
    env.spread_uniforms[0, 0, 1] = 0
    env.step(WAIT)
    assert env.compromised[1] and env.inspection_result[1] == 0 and env.inspection_time[1] == 0


def test_spread_is_simultaneous_not_a_within_step_chain():
    env = controlled_env()
    env.spread_uniforms[:] = 0
    env.step(WAIT)
    assert env.compromised[1] and not env.compromised[2]
    env.step(WAIT)
    assert env.compromised[2]


@pytest.mark.parametrize('patched,decoy,multiplier', [(False, False, 1.), (True, False, .2), (False, True, .25), (True, True, .05)])
def test_defenses_apply_documented_probability_and_leave_residual_risk(patched, decoy, multiplier):
    for factor, expected in ((.5, True), (1.1, False)):
        env = controlled_env()
        env.patched[1], env.decoy[1] = patched, decoy
        threshold = .23 * float(VULNERABILITY[1]) * multiplier
        env.spread_uniforms[0, 0, 1] = threshold * factor
        env.step(WAIT)
        assert bool(env.compromised[1]) == expected


def test_isolation_removes_incoming_and_outgoing_edges():
    outgoing = controlled_env()
    outgoing.spread_uniforms[:] = 0
    outgoing.step(ISOLATE_BASE)
    assert outgoing.compromised.sum() == 1
    incoming = controlled_env()
    incoming.spread_uniforms[:] = 0
    incoming.step(ISOLATE_BASE + 1)
    assert not incoming.compromised[1]
    assert not incoming.action_mask()[PATCH_BASE + 1]
    assert not incoming.action_mask()[INSPECT_BASE + 1]


def test_first_confirmation_reward_cannot_be_farmed():
    env = controlled_env()
    _, _, _, _, first = env.step(INSPECT_BASE)
    _, _, _, _, second = env.step(INSPECT_BASE)
    assert first['reward_terms']['first_confirmation'] == .4
    assert second['reward_terms']['first_confirmation'] == 0
    assert env.first_confirmation_time[0] == 0
    assert first['reward_terms']['action_cost'] == second['reward_terms']['action_cost'] == -.16


def test_detection_delay_penalty_is_explicit_and_ends_at_confirmation_or_isolation():
    env = controlled_env()
    _, _, _, _, before = env.step(WAIT)
    assert before['reward_terms']['undetected_compromise'] == pytest.approx(-.04 * CRITICALITY[0])
    _, _, _, _, after = env.step(INSPECT_BASE)
    assert after['reward_terms']['undetected_compromise'] == 0
    isolated = controlled_env()
    _, _, _, _, info = isolated.step(ISOLATE_BASE)
    assert info['reward_terms']['undetected_compromise'] == 0


def test_false_isolation_has_one_time_and_continuing_cost():
    env = controlled_env()
    _, _, _, _, first = env.step(ISOLATE_BASE + 3)
    _, _, _, _, second = env.step(WAIT)
    assert first['reward_terms']['false_isolation'] == pytest.approx(-(.65 + .85 * CRITICALITY[3]))
    assert first['reward_terms']['clean_isolation_downtime'] == pytest.approx(-.03 * CRITICALITY[3])
    assert second['reward_terms']['false_isolation'] == 0
    assert second['reward_terms']['clean_isolation_downtime'] == first['reward_terms']['clean_isolation_downtime']
    assert env.false_isolations == 1


def test_action_budget_and_invalid_actions_are_accounted():
    env = controlled_env(EnvConfig(response_budget=1))
    env.step(PATCH_BASE)
    assert env.budget == 0
    assert np.flatnonzero(env.action_mask()).tolist() == [WAIT]
    _, _, _, _, info = env.step(PATCH_BASE + 1)
    assert info['invalid_action'] and info['reward_terms']['invalid'] == -.45
    assert env.budget == 0 and env.action_costs == 1


@pytest.mark.parametrize('action', [-1, 25, 1.5, True, '1'])
def test_bad_actions_are_rejected(action):
    with pytest.raises(ValueError):
        controlled_env().step(action)


def test_terminal_and_censoring_semantics():
    env = controlled_env(EnvConfig(horizon=1))
    env.compromised[:] = False
    env.compromised[3] = True
    env.first_compromise_time[:] = -1
    env.first_compromise_time[3] = 0
    _, reward, terminated, truncated, info = env.step(WAIT)
    assert terminated and not truncated
    assert info['reward_terms']['terminal_critical'] == -1.4
    assert info['metrics']['censored_confirmation_delay'] == 1
    assert info['metrics']['confirmed_fraction'] == 0
    assert reward == pytest.approx(sum(info['reward_terms'].values()))
    assert env.metrics()['return'] == pytest.approx(reward)
    with pytest.raises(RuntimeError):
        env.step(WAIT)


def test_common_random_fields_are_independent_of_actions_and_sensor_condition():
    a = SentinelEnv(seed=95)
    b = SentinelEnv(seed=95, config=replace(EnvConfig(), alert_sensitivity=.35))
    np.testing.assert_array_equal(a.spread_uniforms, b.spread_uniforms)
    np.testing.assert_array_equal(a.sensor_uniforms, b.sensor_uniforms)
    a.step(WAIT)
    b.step(PATCH_BASE + 3)
    np.testing.assert_array_equal(a.spread_uniforms, b.spread_uniforms)
    np.testing.assert_array_equal(a.sensor_uniforms, b.sensor_uniforms)


def test_reproducibility_for_identical_seed_and_actions():
    a, b = SentinelEnv(seed=124), SentinelEnv(seed=124)
    for action in (INSPECT_BASE, DECOY_BASE + 3, PATCH_BASE + 5, WAIT):
        oa, ra, da, ta, ia = a.step(action)
        ob, rb, db, tb, ib = b.step(action)
        np.testing.assert_array_equal(oa, ob)
        assert (ra, da, ta) == (rb, db, tb)
        assert ia['reward_terms'] == ib['reward_terms']


@pytest.mark.parametrize('kwargs', [{'horizon': 0}, {'response_budget': 0}, {'patch_multiplier': -1}, {'alert_sensitivity': 2}])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        EnvConfig(**kwargs)
