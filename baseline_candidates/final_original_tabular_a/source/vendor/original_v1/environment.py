"""SentinelRL: a small, synthetic defensive cyber-range POMDP.

This is a teaching simulator. It does not generate packets, exploit real systems,
or connect to a network. All attacks are abstract stochastic transitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

HOSTS = ("Edge Gateway", "Web Server", "App Server", "Database", "Staff Laptop", "Backup Vault")
N_HOSTS = len(HOSTS)
WAIT = 0
INSPECT_BASE = 1
PATCH_BASE = 1 + N_HOSTS
ISOLATE_BASE = 1 + 2 * N_HOSTS
DECOY_BASE = 1 + 3 * N_HOSTS
N_ACTIONS = 1 + 4 * N_HOSTS
FEATURES_PER_HOST = 9
OBS_DIM = N_HOSTS * FEATURES_PER_HOST + 4

# Undirected reachability graph. This is an illustrative topology, not a real network.
EDGES = ((0, 1), (1, 2), (2, 3), (1, 4), (4, 2), (3, 5), (2, 5))
ADJ = np.zeros((N_HOSTS, N_HOSTS), dtype=np.float32)
for _a, _b in EDGES:
    ADJ[_a, _b] = ADJ[_b, _a] = 1.0

VULNERABILITY = np.array([0.40, 0.68, 0.58, 0.82, 0.76, 0.48], dtype=np.float32)
CRITICALITY = np.array([0.45, 0.62, 0.80, 1.00, 0.52, 0.96], dtype=np.float32)

@dataclass(frozen=True)
class EnvConfig:
    horizon: int = 48
    response_budget: int = 6
    attack_base_probability: float = 0.23
    alert_sensitivity: float = 0.55
    false_positive_rate: float = 0.08

class SentinelEnv:
    """A partially observable MDP with an explicit noisy belief/telemetry vector."""
    def __init__(self, seed: int | None = None, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    @property
    def observation_space_shape(self) -> tuple[int, ...]:
        return (OBS_DIM,)

    @property
    def action_space_n(self) -> int:
        return N_ACTIONS

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.budget = int(self.config.response_budget)
        self.compromised = np.zeros(N_HOSTS, dtype=bool)
        self.patched = np.zeros(N_HOSTS, dtype=bool)
        self.isolated = np.zeros(N_HOSTS, dtype=bool)
        self.decoy = np.zeros(N_HOSTS, dtype=bool)
        self.inspected = np.zeros(N_HOSTS, dtype=bool)
        self.confirmed = np.zeros(N_HOSTS, dtype=bool)
        self.belief = np.full(N_HOSTS, 0.10, dtype=np.float32)
        self.alert = np.zeros(N_HOSTS, dtype=bool)
        # An attacker begins with a foothold on either the public gateway or a staff laptop.
        entry = int(self.rng.choice([0, 4], p=[0.62, 0.38]))
        self.compromised[entry] = True
        self.belief[entry] = 0.22
        self.inspected[:] = False
        self.confirmed[:] = False
        self.total_compromises = 1
        self.critical_compromises = int(CRITICALITY[entry] >= 0.8)
        self.first_compromise_time = np.full(N_HOSTS, -1, dtype=np.int32)
        self.first_compromise_time[entry] = 0
        self.first_detection_time = np.full(N_HOSTS, -1, dtype=np.int32)
        self.false_isolations = 0
        self.successful_isolations = 0
        self.action_costs = 0
        self.detected_count = 0
        self.newly_compromised_last = 1
        self.episode_return = 0.0
        self._step_compromise_events = [entry]
        return self._observation(), {"entry_host": entry, "action_mask": self.action_mask()}

    def action_mask(self) -> np.ndarray:
        """True indicates an available action; WAIT is always available."""
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[WAIT] = True
        if self.budget <= 0:
            return mask
        for i in range(N_HOSTS):
            # A newly generated alarm or new telemetry makes re-inspection useful again.
            if (not self.inspected[i]) or self.alert[i]:
                mask[INSPECT_BASE + i] = True
            if not self.patched[i] and not self.isolated[i]:
                mask[PATCH_BASE + i] = True
            if not self.isolated[i]:
                mask[ISOLATE_BASE + i] = True
            if not self.decoy[i] and not self.isolated[i]:
                mask[DECOY_BASE + i] = True
        return mask

    def _observation(self) -> np.ndarray:
        # Per host: belief, alarm, inspected, confirmed compromised, vulnerability,
        # criticality, patched, isolated, decoy. True attack state is never exposed.
        block = np.stack([
            self.belief,
            self.alert.astype(np.float32),
            self.inspected.astype(np.float32),
            self.confirmed.astype(np.float32) * self.inspected.astype(np.float32),
            VULNERABILITY,
            CRITICALITY,
            self.patched.astype(np.float32),
            self.isolated.astype(np.float32),
            self.decoy.astype(np.float32),
        ], axis=1).reshape(-1)
        global_features = np.array([
            self.budget / max(1, self.config.response_budget),
            self.t / max(1, self.config.horizon),
            self.detected_count / N_HOSTS,
            self.newly_compromised_last / N_HOSTS,
        ], dtype=np.float32)
        result = np.concatenate([block.astype(np.float32), global_features]).astype(np.float32)
        assert result.shape == (OBS_DIM,)
        return result

    def _observe_alarms(self) -> tuple[int, int]:
        true_alerts = 0
        false_alerts = 0
        old_alerts = self.alert.copy()
        for i in range(N_HOSTS):
            if self.isolated[i]:
                self.alert[i] = False
                continue
            probability = self.config.alert_sensitivity if self.compromised[i] else self.config.false_positive_rate
            self.alert[i] = bool(self.rng.random() < probability)
            if self.alert[i] and not old_alerts[i]:
                if self.compromised[i]:
                    true_alerts += 1
                    if self.first_detection_time[i] < 0:
                        self.first_detection_time[i] = self.t
                        self.detected_count += 1
                else:
                    false_alerts += 1
            # Approximate per-host Bayes filter. This is a compact belief state rather than
            # a claim of exact inference over the coupled network process.
            prior = float(np.clip(self.belief[i], 0.01, 0.99))
            qd, qf = self.config.alert_sensitivity, self.config.false_positive_rate
            if self.alert[i]:
                post = prior * qd / max(1e-8, prior * qd + (1 - prior) * qf)
            else:
                post = prior * (1 - qd) / max(1e-8, prior * (1 - qd) + (1 - prior) * (1 - qf))
            self.belief[i] = float(np.clip(post, 0.01, 0.99))
        return true_alerts, false_alerts

    def _spread_attacks(self) -> tuple[list[int], int]:
        newly = []
        decoy_triggers = 0
        # Snapshot attackers so newly compromised nodes start spreading next step.
        attackers = np.where(self.compromised & ~self.isolated)[0].tolist()
        for source in attackers:
            for target in np.where(ADJ[source] > 0)[0].tolist():
                if self.compromised[target] or self.isolated[target]:
                    continue
                p = self.config.attack_base_probability * float(VULNERABILITY[target])
                if self.patched[target]:
                    p *= 0.16
                if self.decoy[target]:
                    p *= 0.10
                # Lateral movement is stochastic and abstract; no actual exploit executes.
                if self.rng.random() < p:
                    if self.decoy[target]:
                        decoy_triggers += 1
                        self.alert[target] = True
                        self.belief[target] = max(self.belief[target], 0.75)
                    else:
                        self.compromised[target] = True
                        self.first_compromise_time[target] = self.t
                        self.belief[target] = max(self.belief[target], 0.30)
                        self.inspected[target] = False
                        self.confirmed[target] = False
                        newly.append(target)
        return newly, decoy_triggers

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not isinstance(action, (int, np.integer)) or action < 0 or action >= N_ACTIONS:
            raise ValueError(f"action must be an integer in [0,{N_ACTIONS - 1}]")
        pre_mask = self.action_mask()
        invalid = not bool(pre_mask[action])
        reward = 0.0
        inspected_true = inspected_false = 0
        isolated_true = isolated_false = 0
        used_budget = 0
        if invalid:
            reward -= 0.45
        elif action != WAIT:
            used_budget = 1
            self.budget -= 1
            self.action_costs += 1
            reward -= 0.16
            if INSPECT_BASE <= action < PATCH_BASE:
                i = action - INSPECT_BASE
                self.inspected[i] = True
                self.confirmed[i] = bool(self.compromised[i])
                self.alert[i] = False
                self.belief[i] = 0.97 if self.compromised[i] else 0.02
                if self.compromised[i] and not self.isolated[i]:
                    reward += 1.50
                    inspected_true = 1
                    if self.first_detection_time[i] < 0:
                        self.first_detection_time[i] = self.t
                        self.detected_count += 1
                elif not self.compromised[i]:
                    reward -= 0.18
                    inspected_false = 1
            elif PATCH_BASE <= action < ISOLATE_BASE:
                i = action - PATCH_BASE
                self.patched[i] = True
                # Patching mitigates future entry but does not eradicate an existing attacker.
                reward += 0.12 * float(CRITICALITY[i])
            elif ISOLATE_BASE <= action < DECOY_BASE:
                i = action - ISOLATE_BASE
                self.isolated[i] = True
                self.alert[i] = False
                if self.compromised[i]:
                    reward += 1.10 + 0.55 * float(CRITICALITY[i])
                    isolated_true = 1
                    self.successful_isolations += 1
                    self.belief[i] = max(self.belief[i], 0.90)
                else:
                    reward -= 0.65 + 0.85 * float(CRITICALITY[i])
                    isolated_false = 1
                    self.false_isolations += 1
            elif DECOY_BASE <= action < N_ACTIONS:
                i = action - DECOY_BASE
                self.decoy[i] = True
                reward += 0.08 * float(VULNERABILITY[i])

        self.t += 1
        newly, decoy_triggers = self._spread_attacks()
        self._step_compromise_events = newly
        self.newly_compromised_last = len(newly)
        if newly:
            self.total_compromises += len(newly)
            self.critical_compromises += sum(int(CRITICALITY[i] >= 0.8) for i in newly)
            reward -= sum(2.0 + 4.0 * float(CRITICALITY[i]) for i in newly)
        reward += 0.75 * decoy_triggers
        true_alerts, false_alerts = self._observe_alarms()
        reward += 0.10 * true_alerts - 0.06 * false_alerts
        active_compromised = self.compromised & ~self.isolated
        reward -= float(np.sum(CRITICALITY[active_compromised])) * 0.16
        if self.t >= self.config.horizon:
            reward -= 1.4 * int(np.sum(self.compromised & ~self.isolated & (CRITICALITY >= 0.8)))
        terminated = self.t >= self.config.horizon
        truncated = False
        self.episode_return += float(reward)
        info = {
            "invalid_action": bool(invalid),
            "used_budget": used_budget,
            "new_compromises": len(newly),
            "compromise_hosts": newly,
            "critical_compromises_total": self.critical_compromises,
            "active_compromised_at_end": int(np.sum(self.compromised & ~self.isolated)),
            "inspected_true": inspected_true,
            "inspected_false": inspected_false,
            "isolated_true": isolated_true,
            "isolated_false": isolated_false,
            "false_isolations_total": self.false_isolations,
            "successful_isolations_total": self.successful_isolations,
            "true_alerts": true_alerts,
            "false_alerts": false_alerts,
            "decoy_triggers": decoy_triggers,
            "remaining_budget": self.budget,
            "episode_return": self.episode_return,
        }
        if terminated:
            info["episode_summary"] = self.summary()
        info["action_mask"] = self.action_mask()
        return self._observation(), float(reward), terminated, truncated, info

    def summary(self) -> dict[str, float | int]:
        delays = []
        for i in range(N_HOSTS):
            if self.first_compromise_time[i] >= 0:
                if self.first_detection_time[i] >= 0:
                    delays.append(max(0, int(self.first_detection_time[i] - self.first_compromise_time[i])))
                else:
                    delays.append(self.config.horizon - int(self.first_compromise_time[i]))
        return {
            "episode_return": float(self.episode_return),
            "total_compromises": int(self.total_compromises),
            "critical_compromises": int(self.critical_compromises),
            "active_compromised_at_end": int(np.sum(self.compromised & ~self.isolated)),
            "mean_detection_delay": float(np.mean(delays)) if delays else 0.0,
            "false_isolations": int(self.false_isolations),
            "successful_isolations": int(self.successful_isolations),
            "budget_used": int(self.action_costs),
            "remaining_budget": int(self.budget),
            "detected_hosts": int(self.detected_count),
        }

    def render(self) -> str:
        parts = []
        for i, name in enumerate(HOSTS):
            status = "COMPROMISED" if self.compromised[i] else "clean/unknown"
            if self.isolated[i]:
                status += ", isolated"
            if self.patched[i]:
                status += ", patched"
            parts.append(f"{name}: {status}; belief={self.belief[i]:.2f}; alert={int(self.alert[i])}")
        return "\n".join(parts)
