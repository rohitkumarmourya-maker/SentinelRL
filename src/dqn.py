"""Pure observation-only Double DQN using real PyTorch autograd.

No handcrafted Q prior, demonstrations, teacher, post-hoc action shield or hidden
state. Action masks encode environment availability, not strategic preferences.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import random
import numpy as np
import torch
from torch import nn

from .environment import FEATURES_PER_HOST, N_HOSTS, N_ACTIONS, OBS_DIM


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = .98
    learning_rate: float = 3e-4
    batch_size: int = 64
    replay_capacity: int = 50000
    learning_starts: int = 1000
    train_every: int = 4
    target_sync_updates: int = 500
    epsilon_start: float = 1.
    epsilon_end: float = .05
    epsilon_decay_steps: int = 80000
    hidden_size: int = 128
    grad_clip: float = 10.

    def __post_init__(self):
        if not 0 <= self.gamma <= 1 or self.learning_rate <= 0:
            raise ValueError("invalid discount or learning rate")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("invalid exploration schedule")
        for name in ("batch_size", "replay_capacity", "learning_starts", "train_every", "target_sync_updates", "epsilon_decay_steps", "hidden_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay capacity must cover batch size")


def transform_observation(observation: np.ndarray, mode: str = "history") -> np.ndarray:
    result = np.array(observation, dtype=np.float32, copy=True)
    if mode == "no_memory":
        # Retrained ablation: remove EMA and diagnostic history, retain current alert.
        result[:N_HOSTS * FEATURES_PER_HOST].reshape(N_HOSTS, FEATURES_PER_HOST)[:, 1:4] = 0
    elif mode != "history":
        raise ValueError(f"unknown observation mode: {mode}")
    return result


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, n_actions))

    def forward(self, observation):
        return self.net(observation)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.capacity, self.position, self.size = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        self.s = np.empty((capacity, OBS_DIM), dtype=np.float32)
        self.ns = np.empty_like(self.s)
        self.a = np.empty(capacity, dtype=np.int64)
        self.r = np.empty(capacity, dtype=np.float32)
        self.done = np.empty(capacity, dtype=np.float32)
        self.mask = np.empty((capacity, N_ACTIONS), dtype=bool)

    def __len__(self):
        return self.size

    def add(self, s, action, reward, ns, done, next_mask):
        i = self.position
        self.s[i], self.ns[i] = s, ns
        self.a[i], self.r[i], self.done[i], self.mask[i] = action, reward, done, next_mask
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = self.rng.choice(self.size, batch_size, replace=False)
        return tuple(torch.from_numpy(x[indices]) for x in (self.s, self.a, self.r, self.ns, self.done, self.mask))


def double_dqn_targets(rewards, dones, online_next, target_next, masks, gamma):
    """Online selects a legal action; target evaluates it. Terminal value is zero."""
    next_actions = online_next.masked_fill(~masks, -torch.inf).argmax(dim=1, keepdim=True)
    next_values = target_next.gather(1, next_actions).squeeze(1)
    return rewards + gamma * (1 - dones) * next_values


class DQNAgent:
    def __init__(self, seed: int = 0, config: DQNConfig | None = None, observation_mode: str = "history"):
        self.config = config or DQNConfig()
        self.seed, self.observation_mode = seed, observation_mode
        transform_observation(np.zeros(OBS_DIM), observation_mode)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.rng = np.random.default_rng(seed)
        self.online = QNetwork(hidden_size=self.config.hidden_size)
        self.target = QNetwork(hidden_size=self.config.hidden_size)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.replay = ReplayBuffer(self.config.replay_capacity, seed + 1)
        self.environment_steps = self.updates = 0
        self.initial_parameter_hash = self.parameter_hash()

    def epsilon(self):
        fraction = min(1., self.environment_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    def act(self, observation, mask, explore: bool = False):
        legal = np.flatnonzero(mask)
        if not len(legal):
            raise ValueError("at least one legal action is required")
        if explore and self.rng.random() < self.epsilon():
            return int(self.rng.choice(legal))
        state = transform_observation(observation, self.observation_mode)
        with torch.no_grad():
            q = self.online(torch.from_numpy(state).unsqueeze(0)).squeeze(0).numpy()
        return int(np.argmax(np.where(mask, q, -np.inf)))

    def observe(self, state, action, reward, next_state, done, next_mask):
        self.replay.add(transform_observation(state, self.observation_mode), action, reward,
                        transform_observation(next_state, self.observation_mode), done, next_mask)
        self.environment_steps += 1
        if self.environment_steps < self.config.learning_starts or len(self.replay) < self.config.batch_size or self.environment_steps % self.config.train_every:
            return None
        states, actions, rewards, next_states, dones, masks = self.replay.sample(self.config.batch_size)
        predictions = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            targets = double_dqn_targets(rewards, dones, self.online(next_states), self.target(next_states), masks, self.config.gamma)
        loss = nn.functional.smooth_l1_loss(predictions, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_sync_updates == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {"loss": float(loss.detach()), "gradient_norm": float(grad_norm), "update": self.updates}

    def parameter_hash(self):
        digest = hashlib.sha256()
        for name, value in self.online.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path, metadata: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 2, "obs_dim": OBS_DIM, "n_actions": N_ACTIONS,
                   "seed": self.seed, "config": asdict(self.config), "observation_mode": self.observation_mode,
                   "online": self.online.state_dict(), "target": self.target.state_dict(),
                   "environment_steps": self.environment_steps, "updates": self.updates,
                   "initial_parameter_hash": self.initial_parameter_hash,
                   "parameter_hash": self.parameter_hash(), "metadata": metadata or {}}
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 2 or payload.get("obs_dim") != OBS_DIM or payload.get("n_actions") != N_ACTIONS:
            raise ValueError("incompatible checkpoint; legacy models require original source")
        agent = cls(payload["seed"], DQNConfig(**payload["config"]), payload["observation_mode"])
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload["target"])
        agent.environment_steps, agent.updates = payload["environment_steps"], payload["updates"]
        agent.initial_parameter_hash = payload["initial_parameter_hash"]
        agent.checkpoint_metadata = payload["metadata"]
        if agent.parameter_hash() != payload["parameter_hash"]:
            raise ValueError("checkpoint parameter hash mismatch")
        agent.online.eval()
        return agent
