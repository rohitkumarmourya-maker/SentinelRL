"""Lawful adapter and original comparator interface checks."""
import numpy as np
from src.comparators import Lawful58Adapter, OriginalPolicy, verify_previous_source, verify_original_source
from src.environment import SentinelEnv, INSPECT_BASE


def test_historical_sources_and_runtime_are_not_weakened():
    verify_previous_source();verify_original_source()


def test_adapter_uses_only_public_history_and_zeroes_unavailable_channels():
    env=SentinelEnv(seed=13);observation,_=env.reset(13)
    a,b=Lawful58Adapter(),Lawful58Adapter()
    xa=a.reset(observation)
    env.compromised[:]=~env.compromised
    xb=b.reset(observation.copy())
    np.testing.assert_array_equal(xa,xb)
    assert xa.shape==(58,) and xa[57]==0 and xa[56]==0
    np.testing.assert_array_equal(xa[:54].reshape(6,9)[:,0],a.filter.marginals.astype(np.float32))


def test_legacy_inspection_affordance_is_public_and_subset_of_true_mask():
    env=SentinelEnv(seed=2)
    obs,info=env.reset(2)
    obs,_,_,_,info=env.step(INSPECT_BASE+1)
    mask=Lawful58Adapter.legacy_mask(obs,info['action_mask'])
    assert (~mask|info['action_mask']).all()
    nodes=obs[:54].reshape(6,9)
    assert bool(mask[INSPECT_BASE+1])==bool(nodes[1,0]>.5)


def test_original_checkpoint_interface_emits_legal_action():
    env=SentinelEnv(seed=3);obs,info=env.reset(3)
    for kind in ('original_full_transfer','original_no_stealth_transfer','original_alert','original_threshold','original_patch'):
        policy=OriginalPolicy(kind,17);policy.reset(obs,info['action_mask'])
        assert info['action_mask'][policy.act(obs,info['action_mask'])]


def test_random_policy_stream_is_domain_separated_from_environment():
    from src.comparators import make_comparator
    scenario=600000
    policy=make_comparator('original_random',0,scenario)
    expected_seed=int(np.random.SeedSequence([scenario,0x53454E54]).generate_state(1)[0])
    expected=np.random.default_rng(expected_seed).random(10)
    np.testing.assert_array_equal(policy.agent.rng.random(10),expected)
    assert not np.array_equal(expected,np.random.default_rng(scenario).random(10))
