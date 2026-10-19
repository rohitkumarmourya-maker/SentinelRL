"""A high-performance Dueling Double-DQN with expert demonstrations (DQfD) and multi-step learning."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import random
import numpy as np
import torch
from torch import nn
from .policies import AlertResponderPolicy

@dataclass
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 5e-4
    batch_size: int = 64
    replay_capacity: int = 16000
    learning_starts: int = 120
    train_every: int = 2
    target_sync_steps: int = 250
    tau: float = 0.005
    epsilon_start: float = 0.95
    epsilon_end: float = 0.04
    epsilon_decay_steps: int = 6500
    hidden_size: int = 128
    grad_clip: float = 5.0
    n_step: int = 3
    expert_margin: float = 0.8
    margin_weight: float = 0.8

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        # Base representation layer (keeping self.net[0] for backward compatibility)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        # Dueling Value Stream: V(s) -> scalar
        self.val_head = nn.Sequential(
            nn.Linear(hidden_size, max(32, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(32, hidden_size // 2), 1),
        )
        # Dueling Advantage Stream: A(s, a) -> n_actions
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_size, max(32, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(32, hidden_size // 2), n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        val = self.val_head(feat)
        adv = self.adv_head(feat)
        return val + (adv - adv.mean(dim=-1, keepdim=True))

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
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.online = QNetwork(obs_dim, n_actions, self.config.hidden_size).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, self.config.hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optim = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        
        self.replay = ReplayBuffer(self.config.replay_capacity, seed + 1000)
        self.expert_replay: list[tuple] = []
        self.n_step_buffer: deque = deque(maxlen=self.config.n_step)
        
        self.steps = 0
        self.updates = 0
        self.loss_history: list[float] = []
        self.expert_policy = AlertResponderPolicy()

    def epsilon(self) -> float:
        f = min(1.0, self.steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + f * (self.config.epsilon_end - self.config.epsilon_start)

    def act(self, observation, action_mask, explore: bool = False) -> int:
        legal = np.flatnonzero(np.asarray(action_mask, dtype=bool))
        if len(legal) == 0:
            raise RuntimeError("No valid actions; environment must keep WAIT legal")
        if explore and random.random() < self.epsilon():
            if random.random() < 0.60:
                expert_a = self.expert_policy.act(observation, action_mask)
                if action_mask[expert_a]:
                    return int(expert_a)
            return int(random.choice(legal.tolist()))
        with torch.no_grad():
            x = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(x).squeeze(0).cpu().numpy()
        q[~np.asarray(action_mask, dtype=bool)] = -1e9
        return int(np.argmax(q))

    def add_expert(self, s, a, r, ns, done, next_mask):
        self.expert_replay.append((np.asarray(s, np.float32), int(a), float(r),
                                   np.asarray(ns, np.float32), bool(done), np.asarray(next_mask, bool)))

    def remember(self, s, a, r, ns, done, next_mask):
        n = self.config.n_step
        gamma = self.config.gamma
        self.n_step_buffer.append((s, a, r, ns, done, next_mask))
        if len(self.n_step_buffer) == n:
            R = sum(self.n_step_buffer[i][2] * (gamma ** i) for i in range(n))
            first_s, first_a, _, _, _, _ = self.n_step_buffer[0]
            _, _, _, last_ns, last_done, last_mask = self.n_step_buffer[-1]
            self.replay.add(first_s, first_a, R, last_ns, last_done, last_mask)
        if done:
            while len(self.n_step_buffer) > 1:
                self.n_step_buffer.popleft()
                R = sum(self.n_step_buffer[i][2] * (gamma ** i) for i in range(len(self.n_step_buffer)))
                first_s, first_a, _, _, _, _ = self.n_step_buffer[0]
                _, _, _, last_ns, last_done, last_mask = self.n_step_buffer[-1]
                self.replay.add(first_s, first_a, R, last_ns, last_done, last_mask)
            self.n_step_buffer.clear()

    def learn(self) -> float | None:
        total = len(self.replay) + len(self.expert_replay)
        if total < max(self.config.learning_starts, self.config.batch_size):
            return None
        if self.steps % self.config.train_every != 0:
            return None

        batch_size = self.config.batch_size
        n_expert = 0
        if len(self.expert_replay) > 0 and len(self.replay) >= batch_size // 2:
            n_expert = min(batch_size // 2, len(self.expert_replay))
            n_agent = batch_size - n_expert
            agent_s, agent_a, agent_r, agent_ns, agent_d, agent_m = self.replay.sample(n_agent)
            expert_rows = random.sample(self.expert_replay, n_expert)
            exp_s, exp_a, exp_r, exp_ns, exp_d, exp_m = zip(*expert_rows)
            s = np.concatenate([np.stack(exp_s), agent_s])
            a = np.concatenate([np.asarray(exp_a), agent_a])
            r = np.concatenate([np.asarray(exp_r, np.float32), agent_r])
            ns = np.concatenate([np.stack(exp_ns), agent_ns])
            done = np.concatenate([np.asarray(exp_d, np.float32), agent_d])
            masks = np.concatenate([np.stack(exp_m), agent_m])
        elif len(self.expert_replay) > 0 and len(self.replay) < batch_size // 2:
            expert_rows = random.sample(self.expert_replay, min(batch_size, len(self.expert_replay)))
            n_expert = len(expert_rows)
            exp_s, exp_a, exp_r, exp_ns, exp_d, exp_m = zip(*expert_rows)
            s, a, r, ns, done, masks = np.stack(exp_s), np.asarray(exp_a), np.asarray(exp_r, np.float32), np.stack(exp_ns), np.asarray(exp_d, np.float32), np.stack(exp_m)
        else:
            s, a, r, ns, done, masks = self.replay.sample(min(batch_size, len(self.replay)))
            n_expert = 0

        ts = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        ta = torch.as_tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
        tr = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        tns = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        td = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        tm = torch.as_tensor(masks, dtype=torch.bool, device=self.device)

        q_all = self.online(ts)
        q_current = q_all.gather(1, ta).squeeze(1)

        with torch.no_grad():
            next_online = self.online(tns).masked_fill(~tm, -1e9)
            next_actions = next_online.argmax(dim=1, keepdim=True)
            next_q = self.target(tns).gather(1, next_actions).squeeze(1)
            target = tr + (self.config.gamma ** self.config.n_step) * (1.0 - td) * next_q

        loss = nn.functional.smooth_l1_loss(q_current, target)

        if n_expert > 0:
            margin = torch.ones_like(q_all[:n_expert]) * self.config.expert_margin
            margin.scatter_(1, ta[:n_expert], 0.0)
            margin_target = (q_all[:n_expert] + margin).max(dim=1)[0]
            expert_loss = (margin_target - q_current[:n_expert]).clamp(min=0.0).mean()
            loss = loss + self.config.margin_weight * expert_loss

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.grad_clip)
        self.optim.step()
        self.updates += 1

        tau = self.config.tau
        for target_p, online_p in zip(self.target.parameters(), self.online.parameters()):
            target_p.data.copy_(tau * online_p.data + (1.0 - tau) * target_p.data)

        val = float(loss.detach().cpu().item())
        self.loss_history.append(val)
        return val

    def save(self, path: str, metadata: dict | None = None):
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "obs_dim": self.obs_dim, "n_actions": self.n_actions,
                    "config": self.config.__dict__, "steps": self.steps, "updates": self.updates,
                    "metadata": metadata or {}}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DQNAgent":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg_dict = ckpt.get("config", {})
        valid_keys = DQNConfig.__dataclass_fields__.keys()
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        cfg = DQNConfig(**filtered_cfg)
        agent = cls(int(ckpt["obs_dim"]), int(ckpt["n_actions"]), seed=0, config=cfg, device=device)
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt.get("target", ckpt["online"]))
        agent.steps = int(ckpt.get("steps", 0))
        agent.updates = int(ckpt.get("updates", 0))
        agent.target.eval()
        return agent
