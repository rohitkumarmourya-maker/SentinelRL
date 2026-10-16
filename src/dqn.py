"""A compact masked DQN agent implemented for the SentinelRL simulator."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import random
import numpy as np
import torch
from torch import nn

@dataclass
class DQNConfig:
    gamma: float = 0.97
    learning_rate: float = 7e-4
    batch_size: int = 64
    replay_capacity: int = 16000
    learning_starts: int = 500
    train_every: int = 2
    target_sync_steps: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.06
    epsilon_decay_steps: int = 6500
    hidden_size: int = 128
    grad_clip: float = 5.0

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.data = deque(maxlen=capacity)
        self.rng = random.Random(seed)
    def __len__(self): return len(self.data)
    def add(self, s, a, r, ns, done, next_mask):
        self.data.append((np.asarray(s, np.float32), int(a), float(r), np.asarray(ns, np.float32), bool(done), np.asarray(next_mask, bool)))
    def sample(self, batch_size: int):
        rows = self.rng.sample(self.data, batch_size)
        s, a, r, ns, d, masks = zip(*rows)
        return (np.stack(s), np.asarray(a), np.asarray(r, np.float32), np.stack(ns), np.asarray(d, np.float32), np.stack(masks))

class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, seed: int = 0, config: DQNConfig | None = None, device: str = "cpu"):
        self.config = config or DQNConfig()
        self.device = torch.device(device)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.online = QNetwork(obs_dim, n_actions, self.config.hidden_size).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, self.config.hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optim = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.replay = ReplayBuffer(self.config.replay_capacity, seed + 1000)
        self.n_actions = n_actions
        self.steps = 0
        self.updates = 0
        self.loss_history: list[float] = []

    def epsilon(self) -> float:
        f = min(1.0, self.steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + f * (self.config.epsilon_end - self.config.epsilon_start)

    def act(self, observation, action_mask, explore: bool = False) -> int:
        legal = np.flatnonzero(np.asarray(action_mask, dtype=bool))
        if len(legal) == 0:
            raise RuntimeError("No valid actions; environment must keep WAIT legal")
        if explore and random.random() < self.epsilon():
            return int(random.choice(legal.tolist()))
        with torch.no_grad():
            x = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(x).squeeze(0).cpu().numpy()
        q[~np.asarray(action_mask, dtype=bool)] = -1e9
        return int(np.argmax(q))

    def remember(self, *args): self.replay.add(*args)

    def learn(self) -> float | None:
        if len(self.replay) < max(self.config.learning_starts, self.config.batch_size):
            return None
        if self.steps % self.config.train_every != 0:
            return None
        s, a, r, ns, done, masks = self.replay.sample(self.config.batch_size)
        ts = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        ta = torch.as_tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
        tr = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        tns = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        td = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        tm = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        q_current = self.online(ts).gather(1, ta).squeeze(1)
        with torch.no_grad():
            next_online = self.online(tns).masked_fill(~tm, -1e9)
            next_actions = next_online.argmax(dim=1, keepdim=True)
            next_q = self.target(tns).gather(1, next_actions).squeeze(1)
            target = tr + self.config.gamma * (1.0 - td) * next_q
        loss = nn.functional.smooth_l1_loss(q_current, target)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.grad_clip)
        self.optim.step()
        self.updates += 1
        if self.updates % self.config.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        value = float(loss.detach().cpu().item())
        self.loss_history.append(value)
        return value

    def save(self, path: str, metadata: dict | None = None):
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "obs_dim": self.online.net[0].in_features, "n_actions": self.n_actions,
                    "config": self.config.__dict__, "steps": self.steps, "updates": self.updates,
                    "metadata": metadata or {}}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DQNAgent":
        # weights_only=False is needed for plain configuration metadata in PyTorch 2.6+.
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = DQNConfig(**ckpt.get("config", {}))
        agent = cls(int(ckpt["obs_dim"]), int(ckpt["n_actions"]), seed=0, config=cfg, device=device)
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt.get("target", ckpt["online"]))
        agent.steps = int(ckpt.get("steps", 0)); agent.updates = int(ckpt.get("updates", 0))
        agent.target.eval()
        return agent
