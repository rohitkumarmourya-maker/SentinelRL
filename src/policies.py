"""Evaluation baselines and discretized Tabular Q-Learning RL comparator."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from .environment import (
    WAIT,
    N_HOSTS,
    N_ACTIONS,
    INSPECT_BASE,
    PATCH_BASE,
    ISOLATE_BASE,
    DECOY_BASE,
    FEATURES_PER_HOST,
    VULNERABILITY,
    CRITICALITY,
)


class RandomLegalPolicy:
    name = "Random legal"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        return int(self.rng.choice(np.flatnonzero(mask)))


class PatchPriorityPolicy:
    """Harden critical assets first; isolate only when telemetry is very strong."""

    name = "Patch priority"

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        x = obs[: N_HOSTS * FEATURES_PER_HOST].reshape(N_HOSTS, FEATURES_PER_HOST)
        suspicious = [i for i in range(N_HOSTS) if mask[INSPECT_BASE + i] and x[i, 0] > 0.55]
        if suspicious:
            i = max(suspicious, key=lambda j: x[j, 0] * CRITICALITY[j])
            return INSPECT_BASE + i
        confirmed = [i for i in range(N_HOSTS) if x[i, 2] and x[i, 3] and mask[ISOLATE_BASE + i]]
        if confirmed:
            return ISOLATE_BASE + max(confirmed, key=lambda j: CRITICALITY[j])
        patchable = [i for i in range(N_HOSTS) if mask[PATCH_BASE + i]]
        if patchable:
            i = max(patchable, key=lambda j: CRITICALITY[j] * VULNERABILITY[j])
            return PATCH_BASE + i
        return WAIT


class AlertResponderPolicy:
    """Inspect the riskiest unverified host, isolate confirmed compromise, then harden."""

    name = "Alert responder"

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        x = obs[: N_HOSTS * FEATURES_PER_HOST].reshape(N_HOSTS, FEATURES_PER_HOST)
        confirmed = [
            i
            for i in range(N_HOSTS)
            if x[i, 2] > 0.5 and x[i, 3] > 0.5 and not x[i, 7] and mask[ISOLATE_BASE + i]
        ]
        if confirmed:
            i = max(confirmed, key=lambda j: CRITICALITY[j])
            return ISOLATE_BASE + i
        inspectable = [
            i
            for i in range(N_HOSTS)
            if mask[INSPECT_BASE + i] and (x[i, 1] > 0.5 or x[i, 0] > 0.27)
        ]
        if inspectable:
            i = max(inspectable, key=lambda j: x[j, 0] * (0.5 + CRITICALITY[j]))
            return INSPECT_BASE + i
        decoyable = [i for i in range(N_HOSTS) if mask[DECOY_BASE + i]]
        if decoyable:
            i = max(decoyable, key=lambda j: CRITICALITY[j] * VULNERABILITY[j])
            return DECOY_BASE + i
        patchable = [i for i in range(N_HOSTS) if mask[PATCH_BASE + i]]
        if patchable:
            i = max(patchable, key=lambda j: CRITICALITY[j] * VULNERABILITY[j])
            return PATCH_BASE + i
        return WAIT


class ThresholdDefensePolicy:
    """Threshold policy: quarantine only above a high posterior-risk threshold."""

    name = "Threshold defense"

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        x = obs[: N_HOSTS * FEATURES_PER_HOST].reshape(N_HOSTS, FEATURES_PER_HOST)
        likely = [
            i
            for i in range(N_HOSTS)
            if x[i, 0] >= 0.78 and not x[i, 7] and mask[ISOLATE_BASE + i]
        ]
        if likely:
            return ISOLATE_BASE + max(likely, key=lambda j: CRITICALITY[j] * x[j, 0])
        suspects = [
            i
            for i in range(N_HOSTS)
            if mask[INSPECT_BASE + i] and (x[i, 1] > 0.5 or x[i, 0] > 0.4)
        ]
        if suspects:
            return INSPECT_BASE + max(suspects, key=lambda j: x[j, 0])
        patchable = [i for i in range(N_HOSTS) if mask[PATCH_BASE + i]]
        if patchable:
            i = max(patchable, key=lambda j: CRITICALITY[j] * VULNERABILITY[j])
            return PATCH_BASE + i
        return WAIT


class TabularQLearningAgent:
    """Discretized belief-state Tabular Q-Learning agent for RL algorithm ablation comparison."""

    name = "Tabular Q-Learning"

    def __init__(self, seed: int = 0, alpha: float = 0.18, gamma: float = 0.97):
        self.rng = np.random.default_rng(seed)
        self.alpha = alpha
        self.gamma = gamma
        self.q_table: dict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(N_ACTIONS, dtype=np.float32)
        )

    @staticmethod
    def encode_state(obs: np.ndarray) -> tuple[int, ...]:
        x = obs[: N_HOSTS * FEATURES_PER_HOST].reshape(N_HOSTS, FEATURES_PER_HOST)
        unisolated = x[:, 7] < 0.5
        conf_hosts = [i for i in range(N_HOSTS) if unisolated[i] and x[i, 2] > 0.5 and x[i, 3] > 0.5]
        top_conf = max(conf_hosts, key=lambda j: CRITICALITY[j]) if conf_hosts else -1
        scores = np.where(unisolated, x[:, 0] * (0.5 + CRITICALITY), -1.0)
        top_suspect = int(np.argmax(scores))
        max_b = float(x[top_suspect, 0])
        b_bin = 0 if max_b < 0.16 else (1 if max_b < 0.25 else (2 if max_b < 0.65 else 3))
        has_alert = int(np.any((x[:, 1] > 0.5) & unisolated))
        budget_bin = min(3, int(round(obs[54] * 6.0)))
        newly_flag = int(obs[57] * N_HOSTS > 0.5)
        return (top_conf, top_suspect, b_bin, has_alert, budget_bin, newly_flag)

    def act(self, obs: np.ndarray, mask: np.ndarray, epsilon: float = 0.0) -> int:
        legal = np.flatnonzero(mask)
        if epsilon > 0.0 and self.rng.random() < epsilon:
            if self.rng.random() < 0.55:
                return WAIT
            return int(self.rng.choice(legal))
        s = self.encode_state(obs)
        q = self.q_table[s].copy()
        q[~mask] = -1e9
        return int(np.argmax(q))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        s = self.encode_state(obs)
        ns = self.encode_state(next_obs)
        q_next = 0.0 if done else float(np.max(np.where(next_mask, self.q_table[ns], -1e9)))
        td_target = reward + self.gamma * q_next
        self.q_table[s][action] += self.alpha * (td_target - self.q_table[s][action])
