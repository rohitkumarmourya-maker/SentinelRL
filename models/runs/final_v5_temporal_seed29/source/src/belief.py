"""Exact causal Bayes filter using only public observations and chosen actions.

The model is the published simulator's transition/sensor law. No reward, info
payload, latent simulator state, future random fields, or teacher enters here.
Inspection evidence is applied to the pre-spread state, then defense flags to
spread, then the new noisy sensor readings. Isolated sensors are uninformative.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from .environment import (ADJ, VULNERABILITY, N_HOSTS, FEATURES_PER_HOST,
                          EnvConfig, INSPECT_BASE, PATCH_BASE)

STATE_BITS = ((np.arange(1 << N_HOSTS)[:, None] >> np.arange(N_HOSTS)) & 1).astype(bool)
FEATURE_DIM = 56 + 64 + 6


@lru_cache(maxsize=8192)
def _transition_kernel(flags: tuple, attack: float, patch: float, decoy: float):
    flags = np.array(flags, dtype=bool).reshape(N_HOSTS, 3)
    patched, isolated, decoys = flags.T
    target_p = attack * VULNERABILITY.astype(np.float64)
    target_p *= np.where(patched, patch, 1.) * np.where(decoys, decoy, 1.)
    active = STATE_BITS & ~isolated[None, :]
    n_sources = active.astype(np.int8) @ ADJ.astype(np.int8)
    infection = 1 - (1 - target_p[None, :]) ** n_sources
    infection[:, isolated] = 0.
    infection[STATE_BITS] = 1.
    # Conditional on the previous latent state, attacks on different targets
    # use disjoint independent exogenous random variables.
    kernel = np.prod(np.where(STATE_BITS[None, :, :], infection[:, None, :],
                              1 - infection[:, None, :]), axis=2)
    kernel.setflags(write=False)
    return kernel


class ExactBelief:
    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.probabilities = np.zeros(64, dtype=np.float64)
        self.last_observation = None

    @property
    def marginals(self):
        return np.einsum("i,ij->j", self.probabilities, STATE_BITS, optimize=False)

    def _condition(self, likelihood):
        value = self.probabilities * likelihood
        total = value.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("public history has zero likelihood under the published model")
        self.probabilities = value / total

    def _sense(self, observation):
        nodes = np.asarray(observation)[:54].reshape(N_HOSTS, FEATURES_PER_HOST)
        probability = np.where(STATE_BITS, self.config.alert_sensitivity,
                               self.config.false_positive_rate)
        likelihoods = np.where(nodes[None, :, 0] > .5, probability, 1 - probability)
        likelihoods[:, nodes[:, 7] > .5] = 1.
        self._condition(likelihoods.prod(axis=1))

    def reset(self, observation):
        if float(observation[-1]) != 0.:
            raise ValueError("reset requires the initial public observation")
        self.probabilities = np.zeros(64, dtype=np.float64)
        self.probabilities[1 << 0], self.probabilities[1 << 4] = .62, .38
        self._sense(observation)
        self.last_observation = np.array(observation, copy=True)
        return self.features(observation)

    def transition_matrix(self, observation):
        nodes = np.asarray(observation)[:54].reshape(N_HOSTS, FEATURES_PER_HOST)
        return _transition_kernel(tuple((nodes[:, 6:9] > .5).reshape(-1)),
                                  self.config.attack_base_probability,
                                  self.config.patch_multiplier, self.config.decoy_multiplier)

    def update(self, action, observation):
        if self.last_observation is None:
            raise RuntimeError("reset before update")
        prior_nodes = self.last_observation[:54].reshape(N_HOSTS, FEATURES_PER_HOST)
        nodes = np.asarray(observation)[:54].reshape(N_HOSTS, FEATURES_PER_HOST)
        previous_step = round(float(self.last_observation[-1]) * self.config.horizon)
        next_step = round(float(observation[-1]) * self.config.horizon)
        if next_step != previous_step + 1:
            raise ValueError("filter updates must follow consecutive public steps")
        valid_inspection = (INSPECT_BASE <= action < PATCH_BASE and
                            self.last_observation[-2] > 0 and
                            prior_nodes[action - INSPECT_BASE, 7] < .5)
        if valid_inspection:
            host = action - INSPECT_BASE
            result = nodes[host, 2]
            if result not in (.5, 1.):
                raise ValueError("a valid inspection must reveal a binary snapshot")
            self._condition(STATE_BITS[:, host] == (result == 1.))
        self.probabilities = np.einsum("i,ij->j", self.probabilities, self.transition_matrix(observation), optimize=False)
        self._sense(observation)
        self.last_observation = np.array(observation, copy=True)
        return self.features(observation)

    def features(self, observation):
        return np.concatenate((np.asarray(observation, dtype=np.float32),
                               self.probabilities.astype(np.float32),
                               self.marginals.astype(np.float32))).astype(np.float32)
