import unittest
import numpy as np
from src.belief import ExactBelief, STATE_BITS
from src.environment import SentinelEnv, EnvConfig, VULNERABILITY, N_HOSTS


class BeliefTests(unittest.TestCase):
    def test_reset_bayes_and_no_hidden_access(self):
        env = SentinelEnv(123)
        obs, _ = env.reset(123)
        b = ExactBelief()
        b.reset(obs)
        weights = []
        for host, prior in [(0, .62), (4, .38)]:
            p = np.full(6, .08); p[host] = .55
            weights.append(prior * np.prod(np.where(obs[:54].reshape(6,9)[:,0], p, 1-p)))
        expected = np.array(weights) / sum(weights)
        np.testing.assert_allclose(b.probabilities[[1,16]], expected)
        # Arbitrary hidden-state edits cannot affect a filter fed the same history.
        other = ExactBelief(); other.reset(obs.copy())
        env.compromised[:] = ~env.compromised
        np.testing.assert_array_equal(b.probabilities, other.probabilities)

    def test_transition_monotone_stochastic_and_isolation(self):
        env = SentinelEnv(6); obs, _ = env.reset(6)
        b = ExactBelief(); b.reset(obs)
        kernel = b.transition_matrix(obs)
        np.testing.assert_allclose(kernel.sum(1), 1, atol=1e-14)
        for old in range(64):
            self.assertTrue(np.all(kernel[old, (np.arange(64) & old) != old] == 0))
        marginal = kernel[1] @ STATE_BITS
        self.assertAlmostEqual(marginal[1], .23 * float(VULNERABILITY[1]))
        np.testing.assert_allclose(marginal[[0,2,3,4,5]], [1,0,0,0,0])
        isolated = obs.copy(); isolated.reshape(-1)[:54].reshape(6,9)[0,7] = 1
        np.testing.assert_array_equal(b.transition_matrix(isolated)[1], np.eye(64)[1])

    def test_inspection_conditions_pre_spread_not_post_spread(self):
        env = SentinelEnv(8); obs, _ = env.reset(8)
        b = ExactBelief(); b.reset(obs)
        # Construct lawful clean pre-spread diagnostic on host 1 with active host0.
        b.probabilities[:] = 0; b.probabilities[1] = 1
        next_obs = obs.copy(); nodes = next_obs[:54].reshape(6,9)
        nodes[:,0] = 0; nodes[1,2] = .5; nodes[1,3] = 1/48
        next_obs[-1] = 1/48; next_obs[-2] = 5/6
        b.update(2, next_obs)
        p = .23 * float(VULNERABILITY[1])
        expected = p * .45 / (p * .45 + (1-p) * .92)
        self.assertAlmostEqual(b.marginals[1], expected, places=7)
        self.assertGreater(b.marginals[1], 0)

    def test_all_public_histories_normalize_and_isolated_sensor_ignored(self):
        for seed in range(20):
            env = SentinelEnv(seed); obs, info = env.reset(seed)
            b = ExactBelief(); b.reset(obs)
            rng = np.random.default_rng(seed)
            for _ in range(48):
                action = int(rng.choice(np.flatnonzero(info['action_mask'])))
                obs, _, done, _, info = env.step(action)
                b.update(action, obs)
                self.assertAlmostEqual(b.probabilities.sum(), 1., places=12)
                self.assertTrue(np.all(b.probabilities >= 0))
        c = ExactBelief(); c.probabilities[:] = 1/64
        obs[:54].reshape(6,9)[:,7] = 1
        c._sense(obs)
        np.testing.assert_array_equal(c.probabilities, np.full(64,1/64))


if __name__ == '__main__': unittest.main()
