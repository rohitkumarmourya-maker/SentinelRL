"""Original graph-dueling architecture, retrained on lawful features and unchanged R.

The shared v1 training contract uses gamma1,3-step TD, Huber5 and exact
forced-WAIT suffix compression. Pure mode disables both the handcrafted prior
and action shield. Faithful mode preserves those original decision mechanisms.
No hidden reward shaping, demonstrations or teacher loss is applied.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn
from .comparators import Lawful58Adapter
from vendor.original_v1.dqn import QNetwork
FEATURE_DIM = 58
from .environment import ADJ, EnvConfig, N_ACTIONS


@dataclass(frozen=True)
class GraphConfig:
    gamma: float = 1.
    learning_rate: float = 5e-4
    batch_size: int = 128
    replay_capacity: int = 100000
    learning_starts: int = 1000
    train_every: int = 1
    target_sync_updates: int = 500
    epsilon_start: float = 1.
    epsilon_end: float = .02
    epsilon_decay_steps: int = 300000
    hidden_size: int = 128
    n_step: int = 3
    grad_clip: float = 10.
    huber_beta: float = 5.
    compress_forced_wait: bool = True


class Replay:
    def __init__(self, capacity, seed):
        self.capacity, self.position, self.size = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        self.s = np.empty((capacity, FEATURE_DIM), dtype=np.float32)
        self.ns = np.empty_like(self.s)
        self.a = np.empty(capacity, dtype=np.int64)
        self.r = np.empty(capacity, dtype=np.float32)
        self.discount = np.empty(capacity, dtype=np.float32)
        self.mask = np.empty((capacity, N_ACTIONS), dtype=bool)

    def add(self, s, a, r, ns, discount, mask):
        i = self.position
        self.s[i], self.a[i], self.r[i] = s, a, r
        self.ns[i], self.discount[i], self.mask[i] = ns, discount, mask
        self.position = (i+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, n):
        index = self.rng.integers(self.size, size=n)
        return tuple(torch.from_numpy(x[index]) for x in (self.s, self.a, self.r, self.ns, self.discount, self.mask))


class GraphBaselineAgent:
    def __init__(self, seed=0, config: GraphConfig | None = None, env_config: EnvConfig | None = None, kind: str = "pure"):
        self.seed, self.config = seed, config or GraphConfig()
        if kind not in ("pure","faithful"):
            raise ValueError(kind)
        self.kind = kind
        self.env_config = env_config or EnvConfig()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.online = QNetwork(58, N_ACTIONS, hidden_size=self.config.hidden_size, use_topological_prior=(kind == "faithful"), use_stealth_detector=False)
        self.target = QNetwork(58, N_ACTIONS, hidden_size=self.config.hidden_size, use_topological_prior=(kind == "faithful"), use_stealth_detector=False)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.replay = Replay(self.config.replay_capacity, seed+1)
        self.environment_steps = self.updates = self.decision_transitions = 0
        self.initial_parameter_hash = self.parameter_hash()
        self.adapter = Lawful58Adapter(self.env_config)
        self.current_features = self.last_action = self.last_time = None
        self.trajectory = []
        self.checkpoint_metadata = {}

    def reset(self, observation, mask=None):
        self.current_features = self.adapter.reset(observation)
        self.last_action, self.last_time = None, 0
        self.trajectory = []
        return self.current_features

    def _prepare(self, observation):
        current_time = round(float(observation[-1]) * self.env_config.horizon)
        if self.last_time is None or current_time == 0 and self.last_time != 0:
            self.reset(observation)
        elif current_time != self.last_time:
            self.adapter.previous_action = self.last_action
            self.current_features = self.adapter.advance(observation)
            self.last_time = current_time
        return self.current_features

    def epsilon(self):
        c = self.config
        return c.epsilon_start + min(1., self.environment_steps/c.epsilon_decay_steps) * (c.epsilon_end-c.epsilon_start)

    def act(self, observation, mask, explore=False):
        features = self._prepare(observation)
        mask = self.adapter.legacy_mask(observation, mask) if self.kind == 'faithful' else np.array(mask, copy=True)
        legal = np.flatnonzero(mask)
        if len(legal) == 1:
            action = int(legal[0])
        elif explore and self.rng.random() < self.epsilon():
            action = int(self.rng.choice(legal))
        else:
            with torch.no_grad():
                q = self.online(torch.from_numpy(features)[None]).numpy()[0].copy()
            if self.kind == 'faithful':
                nodes = features[:54].reshape(6,9)
                for host in range(6):
                    isolate = 13+host
                    if mask[isolate] and not (nodes[host,2] > .5 and nodes[host,3] > .5) and nodes[host,0] < .70:
                        q[isolate] -= 10.
            action = int(np.argmax(np.where(mask,q,-np.inf)))
        self.acted_features = features.copy()
        self.acted_mask = np.array(mask,copy=True)
        self.last_action = action
        return action

    def observe(self, state, action, reward, next_state, done, next_mask):
        next_mask = self.adapter.legacy_mask(next_state,next_mask) if self.kind == 'faithful' else next_mask
        features = self.acted_features
        next_features = self._prepare(next_state).copy()
        self.environment_steps += 1
        if self.config.compress_forced_wait and self.acted_mask.sum() == 1 and self.trajectory:
            previous = self.trajectory[-1]
            previous[2] += self.config.gamma ** previous[6] * reward
            previous[3], previous[4], previous[5] = next_features, done, np.array(next_mask, copy=True)
            previous[6] += 1
        else:
            self.trajectory.append([features, action, reward, next_features, done, np.array(next_mask, copy=True), 1])
        if done:
            return self._finish_episode()
        return None

    def _finish_episode(self):
        c = self.config
        for start in range(len(self.trajectory)):
            total, duration = 0., 0
            end = start
            for end in range(start, min(start+c.n_step, len(self.trajectory))):
                transition = self.trajectory[end]
                total += c.gamma ** duration * transition[2]
                duration += transition[6]
                if transition[4]:
                    break
            first, last = self.trajectory[start], self.trajectory[end]
            self.replay.add(first[0], first[1], total, last[3],
                            0. if last[4] else c.gamma ** duration, last[5])
        n_decisions = len(self.trajectory)
        self.decision_transitions += n_decisions
        losses = []
        gradients = []
        if self.replay.size >= max(c.batch_size, c.learning_starts):
            for _ in range(max(1, n_decisions // c.train_every)):
                states, actions, rewards, next_states, discounts, masks = self.replay.sample(c.batch_size)
                predictions = self.online(states).gather(1, actions[:,None]).squeeze(1)
                with torch.no_grad():
                    choices = self.online(next_states).masked_fill(~masks, -torch.inf).argmax(1, keepdim=True)
                    values = self.target(next_states).gather(1, choices).squeeze(1)
                    targets = rewards + discounts * values
                loss = nn.functional.smooth_l1_loss(predictions, targets, beta=c.huber_beta)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), c.grad_clip)
                self.optimizer.step()
                self.updates += 1
                if self.updates % c.target_sync_updates == 0:
                    self.target.load_state_dict(self.online.state_dict())
                losses.append(float(loss.detach()))
                gradients.append(float(norm))
        self.trajectory = []
        return {"loss": float(np.mean(losses)) if losses else 0.,
                "gradient_norm": float(np.mean(gradients)) if gradients else 0.,
                "updates": self.updates, "decision_transitions": self.decision_transitions}

    def parameter_hash(self):
        digest = hashlib.sha256()
        for name, value in self.online.state_dict().items():
            digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def save(self, path, metadata=None):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 4, "algorithm": "original_graph_dueling_retrained", "kind": self.kind,
                    "feature_dim": FEATURE_DIM, "seed": self.seed, "config": asdict(self.config),
                    "environment": asdict(self.env_config), "online": self.online.state_dict(),
                    "target": self.target.state_dict(), "environment_steps": self.environment_steps,
                    "updates": self.updates, "decision_transitions": self.decision_transitions,
                    "initial_parameter_hash": self.initial_parameter_hash,
                    "parameter_hash": self.parameter_hash(), "metadata": metadata or {}}, path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 4 or payload.get("feature_dim") != FEATURE_DIM:
            raise ValueError("incompatible retrained graph checkpoint")
        agent = cls(payload['seed'], GraphConfig(**payload['config']), EnvConfig(**payload['environment']), payload['kind'])
        agent.online.load_state_dict(payload['online']); agent.target.load_state_dict(payload['target'])
        for name in ('environment_steps', 'updates', 'decision_transitions', 'initial_parameter_hash'):
            setattr(agent, name, payload[name])
        agent.checkpoint_metadata = payload['metadata']
        if agent.parameter_hash() != payload['parameter_hash']:
            raise ValueError("checkpoint parameter hash mismatch")
        agent.online.eval()
        return agent
