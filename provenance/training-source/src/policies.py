"""Observation-only controls and a coarse tabular Q-learning RL baseline.

The handcrafted alert responder is a diagnostic comparator, never a DQN teacher.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .environment import (N_HOSTS, FEATURES_PER_HOST, WAIT, INSPECT_BASE, PATCH_BASE,
                          ISOLATE_BASE, DECOY_BASE, N_ACTIONS)


class NoOpPolicy:
    def act(self, observation, mask, explore=False):
        return WAIT


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, observation, mask, explore=False):
        return int(self.rng.choice(np.flatnonzero(mask)))


class AlertResponderPolicy:
    """Fixed sensor rule used only to measure whether learned control adds value."""
    def act(self, observation, mask, explore=False):
        nodes = observation[:N_HOSTS * FEATURES_PER_HOST].reshape(N_HOSTS, FEATURES_PER_HOST)
        alarm, ema, diagnosis, age, vuln, criticality = nodes[:, :6].T
        candidates = (diagnosis == 1) & mask[ISOLATE_BASE:DECOY_BASE]
        if candidates.any():
            host = int(np.argmax(np.where(candidates, criticality, -1)))
            return ISOLATE_BASE + host
        # Do not immediately repeat a clean diagnostic; it remains a stale result.
        candidates = mask[INSPECT_BASE:PATCH_BASE] & ((diagnosis == 0) | (age >= 3 / 48))
        score = .5 * alarm + ema
        candidates &= score >= .45
        if candidates.any():
            host = int(np.argmax(np.where(candidates, score * (1 + criticality), -1)))
            return INSPECT_BASE + host
        return WAIT


class TabularQAgent:
    """Actual one-step Q-learning over a deliberately coarse observation encoding."""
    def __init__(self, seed=0, gamma=.98, alpha=.15, decay_steps=80000):
        self.seed, self.gamma, self.alpha, self.decay_steps = seed, gamma, alpha, decay_steps
        self.rng = np.random.default_rng(seed)
        self.table = {}
        self.environment_steps = self.updates = 0

    @staticmethod
    def key(observation):
        nodes = observation[:N_HOSTS * FEATURES_PER_HOST].reshape(N_HOSTS, FEATURES_PER_HOST)
        # Fixed v/c omitted: host positions preserve identities. Quantized sensor
        # and own defense history only; no true compromised-state bits.
        values = np.column_stack((nodes[:, 0], np.rint(nodes[:, 2] * 2), nodes[:, 6:9])).astype(int)
        return ','.join(map(str, values.reshape(-1))) + f'|{int(round(observation[-2] * 6))}|{min(3, int(observation[-1] * 4))}'

    def epsilon(self):
        return 1. - .95 * min(1., self.environment_steps / self.decay_steps)

    def values(self, observation):
        return self.table.get(self.key(observation), np.zeros(N_ACTIONS, dtype=np.float64))

    def act(self, observation, mask, explore=False):
        legal = np.flatnonzero(mask)
        if explore and self.rng.random() < self.epsilon():
            return int(self.rng.choice(legal))
        return int(np.argmax(np.where(mask, self.values(observation), -np.inf)))

    def observe(self, state, action, reward, next_state, done, next_mask):
        key = self.key(state)
        values = self.table.setdefault(key, np.zeros(N_ACTIONS, dtype=np.float64))
        continuation = 0. if done else float(np.max(self.values(next_state)[next_mask]))
        delta = reward + self.gamma * continuation - values[action]
        values[action] += self.alpha * delta
        self.environment_steps += 1
        self.updates += 1
        return {"td_error": float(delta), "update": self.updates}

    def save(self, path, metadata=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"format_version": 2, "algorithm": "tabular_q", "seed": self.seed,
              "gamma": self.gamma, "alpha": self.alpha, "decay_steps": self.decay_steps,
              "environment_steps": self.environment_steps, "updates": self.updates,
              "table": {key: value.tolist() for key, value in self.table.items()}, "metadata": metadata or {}}))

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload.get("format_version") != 2 or payload.get("algorithm") != "tabular_q":
            raise ValueError("incompatible tabular checkpoint")
        agent = cls(payload["seed"], payload["gamma"], payload["alpha"], payload["decay_steps"])
        agent.table = {key: np.array(value, dtype=np.float64) for key, value in payload["table"].items()}
        agent.checkpoint_metadata = payload["metadata"]
        agent.environment_steps, agent.updates = payload["environment_steps"], payload["updates"]
        return agent
