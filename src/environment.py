"""Standalone finite-horizon cyber-defense POMDP; no real network operations.

Policies receive only observation and action_mask(). The latent simulator state and
privileged diagnostics in info are for reward/evaluation, never policy inputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import numpy as np

HOSTS = ("Edge Gateway", "Web Server", "App Server", "Database", "Staff Laptop", "Backup Vault")
N_HOSTS = 6
WAIT, INSPECT_BASE, PATCH_BASE, ISOLATE_BASE, DECOY_BASE = 0, 1, 7, 13, 19
N_ACTIONS, FEATURES_PER_HOST, OBS_DIM = 25, 9, 56
EDGES = ((0, 1), (1, 2), (2, 3), (1, 4), (4, 2), (3, 5), (2, 5))
ADJ = np.zeros((N_HOSTS, N_HOSTS), dtype=bool)
for _a, _b in EDGES:
    ADJ[_a, _b] = ADJ[_b, _a] = True
VULNERABILITY = np.array([.40, .68, .58, .82, .76, .48], dtype=np.float32)
CRITICALITY = np.array([.45, .62, .80, 1.00, .52, .96], dtype=np.float32)
CRITICAL = CRITICALITY >= .8
OBS_FEATURES = ("alert", "alert_ema", "inspection_result", "inspection_age", "vulnerability", "criticality", "patched", "isolated", "decoy")


@dataclass(frozen=True)
class EnvConfig:
    horizon: int = 48
    response_budget: int = 6
    attack_base_probability: float = .23
    alert_sensitivity: float = .55
    false_positive_rate: float = .08
    patch_multiplier: float = .20
    decoy_multiplier: float = .25
    alert_ema_alpha: float = .35

    def __post_init__(self):
        if self.horizon < 1 or self.response_budget < 1:
            raise ValueError("horizon and response_budget must be positive")
        for name in ("attack_base_probability", "alert_sensitivity", "false_positive_rate", "patch_multiplier", "decoy_multiplier", "alert_ema_alpha"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0,1]")


class SentinelEnv:
    """Six-host POMDP with explicit observation history and action-independent noise.

    Hidden state: compromised hosts and first-event accounting. Public state:
    time, budget, own defense flags, noisy alerts and explicit inspection history.
    An inspection is a perfect snapshot before this step's spread; it may become
    stale. No unseen infection changes that history or the availability mask.
    """
    def __init__(self, seed: int | None = None, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.reset(seed)

    @property
    def observation_space_shape(self):
        return (OBS_DIM,)

    @property
    def action_space_n(self):
        return N_ACTIONS

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        rng = np.random.default_rng(seed)
        self.seed = seed
        # Draw complete exogenous random fields irrespective of chosen actions.
        self.entry_uniform = float(rng.random())
        self.spread_uniforms = rng.random((self.config.horizon, N_HOSTS, N_HOSTS))
        self.sensor_uniforms = rng.random((self.config.horizon + 1, N_HOSTS))
        self.t, self.budget = 0, self.config.response_budget
        self.compromised = np.zeros(N_HOSTS, dtype=bool)
        self.compromised[0 if self.entry_uniform < .62 else 4] = True
        self.patched = np.zeros(N_HOSTS, dtype=bool)
        self.isolated = np.zeros(N_HOSTS, dtype=bool)
        self.decoy = np.zeros(N_HOSTS, dtype=bool)
        self.inspection_result = np.full(N_HOSTS, -1, dtype=np.int8)
        self.inspection_time = np.full(N_HOSTS, -1, dtype=np.int32)
        self.first_compromise_time = np.where(self.compromised, 0, -1).astype(np.int32)
        self.first_confirmation_time = np.full(N_HOSTS, -1, dtype=np.int32)
        self.alert = np.zeros(N_HOSTS, dtype=bool)
        self.alert_ema = np.zeros(N_HOSTS, dtype=np.float32)
        self.false_isolations = self.successful_isolations = self.action_costs = 0
        self.invalid_actions = 0
        self.action_counts = np.zeros(N_ACTIONS, dtype=np.int32)
        self.episode_return = 0.
        self._sense()
        return self._observation(), {"action_mask": self.action_mask()}

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[WAIT] = True
        if self.budget > 0:
            available = ~self.isolated
            mask[INSPECT_BASE:PATCH_BASE] = available
            mask[PATCH_BASE:ISOLATE_BASE] = available & ~self.patched
            mask[ISOLATE_BASE:DECOY_BASE] = available
            mask[DECOY_BASE:] = available & ~self.decoy
        return mask

    def _observation(self) -> np.ndarray:
        # Each term below is known to the defender. Do not add latent metrics here.
        age = np.where(self.inspection_time >= 0, self.t - self.inspection_time, self.config.horizon)
        nodes = np.stack((self.alert, self.alert_ema, (self.inspection_result + 1) / 2,
                          age / self.config.horizon, VULNERABILITY, CRITICALITY,
                          self.patched, self.isolated, self.decoy), axis=1)
        globals_ = [self.budget / self.config.response_budget, self.t / self.config.horizon]
        return np.concatenate((nodes.reshape(-1), globals_)).astype(np.float32)

    def _sense(self) -> None:
        probabilities = np.where(self.compromised, self.config.alert_sensitivity, self.config.false_positive_rate)
        self.alert = (self.sensor_uniforms[self.t] < probabilities) & ~self.isolated
        alpha = self.config.alert_ema_alpha
        self.alert_ema = ((1 - alpha) * self.alert_ema + alpha * self.alert).astype(np.float32)

    def _spread(self) -> np.ndarray:
        active_sources = self.compromised & ~self.isolated
        susceptible = ~self.compromised & ~self.isolated
        probability = self.config.attack_base_probability * VULNERABILITY
        probability = probability * np.where(self.patched, self.config.patch_multiplier, 1.)
        probability = probability * np.where(self.decoy, self.config.decoy_multiplier, 1.)
        attempts = (self.spread_uniforms[self.t] < probability[None, :])
        attempts &= ADJ & active_sources[:, None] & susceptible[None, :]
        return attempts.any(axis=0)  # simultaneous; new attackers spread next step

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.t >= self.config.horizon:
            raise RuntimeError("episode ended; reset before stepping")
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < N_ACTIONS:
            raise ValueError(f"action must be an integer in [0,{N_ACTIONS - 1}]")
        terms = {key: 0. for key in ("action_cost", "invalid", "first_confirmation", "new_compromise", "active_compromise", "undetected_compromise", "false_isolation", "clean_isolation_downtime", "terminal_critical")}
        invalid = not self.action_mask()[action]
        self.action_counts[action] += 1
        if invalid:
            terms["invalid"] = -.45
            self.invalid_actions += 1
        elif action != WAIT:
            self.budget -= 1
            self.action_costs += 1
            terms["action_cost"] = -.16
            if INSPECT_BASE <= action < PATCH_BASE:
                host = action - INSPECT_BASE
                self.inspection_time[host] = self.t
                self.inspection_result[host] = int(self.compromised[host])
                if self.compromised[host] and self.first_confirmation_time[host] < 0:
                    self.first_confirmation_time[host] = self.t
                    terms["first_confirmation"] = .4
            elif PATCH_BASE <= action < ISOLATE_BASE:
                self.patched[action - PATCH_BASE] = True
            elif ISOLATE_BASE <= action < DECOY_BASE:
                host = action - ISOLATE_BASE
                self.isolated[host] = True
                if self.compromised[host]:
                    self.successful_isolations += 1
                else:
                    self.false_isolations += 1
                    terms["false_isolation"] = -(.65 + .85 * float(CRITICALITY[host]))
            else:
                self.decoy[action - DECOY_BASE] = True
        new = self._spread()
        self.t += 1
        self.compromised |= new
        self.first_compromise_time[new] = self.t
        active = self.compromised & ~self.isolated
        terms["new_compromise"] = -float(np.sum(2 + 4 * CRITICALITY[new]))
        terms["active_compromise"] = -.16 * float(CRITICALITY[active].sum())
        terms["undetected_compromise"] = -.04 * float(CRITICALITY[active & (self.first_confirmation_time < 0)].sum())
        terms["clean_isolation_downtime"] = -.03 * float(CRITICALITY[self.isolated & ~self.compromised].sum())
        terminated = self.t == self.config.horizon
        if terminated:
            terms["terminal_critical"] = -1.4 * int(np.sum(active & CRITICAL))
        self._sense()
        reward = float(sum(terms.values()))
        self.episode_return += reward
        info = {"action_mask": self.action_mask(), "reward_terms": terms,
                "invalid_action": bool(invalid)}
        if terminated:
            info["metrics"] = self.metrics()  # diagnostics, never policy inputs
        return self._observation(), reward, terminated, False, info

    def metrics(self) -> dict[str, float | int]:
        infected = self.first_compromise_time >= 0
        confirmed = self.first_confirmation_time >= 0
        # Right-censored delay: unconfirmed infection has delay to episode end.
        end_times = np.where(confirmed, self.first_confirmation_time, self.t)
        delays = (end_times - self.first_compromise_time)[infected]
        return {"return": float(self.episode_return), "total_compromises": int(infected.sum()),
                "critical_compromises": int(np.sum(infected & CRITICAL)),
                "uncontained_at_end": int(np.sum(self.compromised & ~self.isolated)),
                "false_isolations": self.false_isolations,
                "successful_isolations": self.successful_isolations,
                "action_costs": self.action_costs, "invalid_actions": self.invalid_actions,
                "confirmed_fraction": float(confirmed.sum() / max(1, infected.sum())),
                "censored_confirmation_delay": float(delays.mean()) if len(delays) else 0.}

    def public_snapshot(self) -> dict[str, Any]:
        """Environment-only demonstration; deliberately omits actual attacker state."""
        return {"time": self.t, "budget": self.budget,
                "hosts": [{"name": HOSTS[i], "alert": bool(self.alert[i]),
                           "last_inspection": ("unknown", "clean", "compromised")[int(self.inspection_result[i]) + 1],
                           "patched": bool(self.patched[i]), "isolated": bool(self.isolated[i]),
                           "decoy": bool(self.decoy[i])} for i in range(N_HOSTS)]}


def action_name(action: int) -> str:
    if action == WAIT:
        return "wait"
    kind, host = divmod(action - 1, N_HOSTS)
    return f"{('inspect', 'patch', 'isolate', 'decoy')[kind]}:{HOSTS[host]}"
