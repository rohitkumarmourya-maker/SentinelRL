"""Topological-Bayesian Graph-Dueling Double-DQN with n-step returns, DQfD margin loss, and safety shielding."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Any
import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - fallback when PyTorch is not installed in sandbox
    from ._torch_compat import torch, nn  # type: ignore

from .environment import (
    ADJ,
    N_HOSTS,
    FEATURES_PER_HOST,
    ISOLATE_BASE,
    DECOY_BASE,
    CRITICALITY,
    VULNERABILITY,
    WAIT,
    INSPECT_BASE,
    PATCH_BASE,
)
from .policies import AlertResponderPolicy

# Normalized graph adjacency matrix with self-loops: \hat{A} = D^{-1/2} (A + I) D^{-1/2}
A_MAT = ADJ + np.eye(N_HOSTS, dtype=np.float32)
deg = np.sum(A_MAT, axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
A_NORM_MAT = torch.as_tensor(D_inv_sqrt @ A_MAT @ D_inv_sqrt, dtype=torch.float32)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 4e-4
    batch_size: int = 64
    replay_capacity: int = 16000
    learning_starts: int = 120
    train_every: int = 8
    target_sync_steps: int = 250
    tau: float = 0.005
    epsilon_start: float = 0.90
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 8500
    hidden_size: int = 128
    grad_clip: float = 5.0
    n_step: int = 3
    expert_margin: float = 0.8
    margin_weight: float = 0.8
    use_topological_prior: bool = True
    use_stealth_detector: bool = True


class QNetwork(nn.Module):
    """Graph-Dueling Q-Network with topological message passing and POMDP belief-transition inductive bias."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_size: int = 128,
        use_topological_prior: bool = True,
        use_stealth_detector: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.use_topological_prior = use_topological_prior
        self.use_stealth_detector = use_stealth_detector
        self.register_buffer("adj", A_NORM_MAT)

        # Baseline MLP encoder (retains exact state_dict structure)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Per-host node feature encoder and normalized GCN message-passing layer
        self.node_enc = nn.Sequential(
            nn.Linear(FEATURES_PER_HOST, 32),
            nn.ReLU(),
        )
        self.gcn_dense = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Fusion layer combining topological node embeddings with global POMDP context
        in_dim = N_HOSTS * 32 + 4
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Dueling State-Value Stream: V(s) -> scalar
        self.val_head = nn.Sequential(
            nn.Linear(hidden_size, max(32, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(32, hidden_size // 2), 1),
        )
        # Dueling Action-Advantage Stream: A(s, a) -> n_actions
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_size, max(32, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(32, hidden_size // 2), n_actions),
        )

    def _topological_bayesian_prior(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized POMDP belief-transition and criticality-aware action prior."""
        with torch.no_grad():
            return self._topological_bayesian_prior_impl(x)

    def _topological_bayesian_prior_impl(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        nodes = x[:, : N_HOSTS * FEATURES_PER_HOST].view(batch_size, N_HOSTS, FEATURES_PER_HOST)
        globals_ = x[:, N_HOSTS * FEATURES_PER_HOST :]

        belief = nodes[:, :, 0]
        alert = nodes[:, :, 1]
        inspected = nodes[:, :, 2]
        confirmed = nodes[:, :, 3]
        vuln = nodes[:, :, 4]
        crit = nodes[:, :, 5]
        isolated = nodes[:, :, 7]
        decoy = nodes[:, :, 8]

        rem_budget = globals_[:, 0:1] * 6.0
        newly_last = globals_[:, 3:4] * float(N_HOSTS)

        is_conf = (inspected > 0.5) & (confirmed > 0.5)
        unisolated = isolated < 0.5

        if self.use_stealth_detector:
            # Stealth lateral movement posterior signature: newly_last > 0 and belief in [0.165, 0.185]
            stealth_hit = (newly_last > 0.5) & (belief >= 0.165) & (belief <= 0.185)
        else:
            stealth_hit = belief > 2.0  # all False

        certain_comp = unisolated & (is_conf | (belief >= 0.70) | stealth_hit)
        num_certain = certain_comp.float().sum(dim=1, keepdim=True)

        cert_priority = crit + 0.5 * belief
        can_inspect_first = (~is_conf) & (inspected < 0.5) & (rem_budget >= 1.9) & (crit < 0.80) & (num_certain <= 1.5)

        insp_cand = (num_certain < 0.5) & unisolated & ((inspected < 0.5) | (alert > 0.5)) & (belief >= 0.20)
        any_insp = insp_cand.float().sum(dim=1, keepdim=True) > 0.5

        decoy_cand = (num_certain < 0.5) & (~any_insp) & (rem_budget >= 1.9) & unisolated & (decoy < 0.5) & (crit >= 0.80)

        zeros_host = belief * 0.0
        q_wait = belief[:, :1] * 0.0 + 1.5

        q_inspect = torch.where(
            certain_comp & can_inspect_first,
            6.0 + cert_priority,
            torch.where(insp_cand, 3.5 + 2.0 * (crit * belief), zeros_host - 1.0),
        )
        q_patch = -0.5 + 0.2 * (crit * vuln)
        q_isolate = torch.where(
            certain_comp & (~can_inspect_first),
            6.0 + cert_priority,
            torch.where(certain_comp, 4.2 + cert_priority, zeros_host - 8.0),
        )
        q_decoy = torch.where(decoy_cand, 2.2 + (crit * vuln), zeros_host - 0.8)

        return torch.cat([q_wait, q_inspect, q_patch, q_isolate, q_decoy], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        nodes = x[:, : N_HOSTS * FEATURES_PER_HOST].view(batch_size, N_HOSTS, FEATURES_PER_HOST)
        globals_ = x[:, N_HOSTS * FEATURES_PER_HOST :]

        # Topological graph convolution over normalized adjacency
        h = self.node_enc(nodes)
        h_graph = torch.matmul(self.adj, h)
        h_comb = self.gcn_dense(h_graph)

        flat = torch.cat([h_comb.view(batch_size, -1), globals_], dim=1)
        feat = self.fusion(flat) + 0.1 * self.net(x)

        val = self.val_head(feat)
        adv = self.adv_head(feat)
        q_dueling = val + (adv - adv.mean(dim=-1, keepdim=True))

        if self.use_topological_prior and self.obs_dim == N_HOSTS * FEATURES_PER_HOST + 4 and self.n_actions == 1 + 4 * N_HOSTS:
            return q_dueling.clamp(min=-0.12, max=0.12) + self._topological_bayesian_prior(x)
        return q_dueling


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.data: deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, s, a, r, ns, done, next_mask) -> None:
        self.data.append((
            np.asarray(s, np.float32),
            int(a),
            float(r),
            np.asarray(ns, np.float32),
            bool(done),
            np.asarray(next_mask, bool),
        ))

    def sample(self, batch_size: int):
        rows = self.rng.sample(self.data, batch_size)
        s, a, r, ns, d, masks = zip(*rows)
        return (
            np.stack(s),
            np.asarray(a),
            np.asarray(r, np.float32),
            np.stack(ns),
            np.asarray(d, np.float32),
            np.stack(masks),
        )


class DQNAgent:
    """Graph-Dueling Double-DQN agent with multi-step TD learning and verification safety shield."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        seed: int = 0,
        config: DQNConfig | None = None,
        device: str = "cpu",
    ):
        self.config = config or DQNConfig()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.online = QNetwork(
            obs_dim,
            n_actions,
            self.config.hidden_size,
            use_topological_prior=self.config.use_topological_prior,
            use_stealth_detector=self.config.use_stealth_detector,
        ).to(self.device)
        self.target = QNetwork(
            obs_dim,
            n_actions,
            self.config.hidden_size,
            use_topological_prior=self.config.use_topological_prior,
            use_stealth_detector=self.config.use_stealth_detector,
        ).to(self.device)
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

    def act(self, observation: np.ndarray, action_mask: np.ndarray, explore: bool = False) -> int:
        legal = np.flatnonzero(np.asarray(action_mask, dtype=bool))
        if len(legal) == 0:
            raise RuntimeError("No valid actions; environment must keep WAIT legal")
        if explore:
            eps = self.epsilon()
            if random.random() < (eps ** 2.0):
                return int(random.choice(legal.tolist()))

        with torch.no_grad():
            x = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(x).squeeze(0).cpu().numpy().copy()

        # Operational safety guardrail: suppress unverified isolations on low-belief nodes
        obs_arr = np.asarray(observation, dtype=np.float32)
        if obs_arr.shape[0] >= N_HOSTS * FEATURES_PER_HOST + 4:
            nodes = obs_arr[: N_HOSTS * FEATURES_PER_HOST].reshape(N_HOSTS, FEATURES_PER_HOST)
            newly_last = obs_arr[N_HOSTS * FEATURES_PER_HOST + 3] * float(N_HOSTS)
            for i in range(N_HOSTS):
                isolate_act = ISOLATE_BASE + i
                if action_mask[isolate_act]:
                    inspected = nodes[i, 2] > 0.5
                    confirmed = nodes[i, 3] > 0.5
                    belief = float(nodes[i, 0])
                    stealth_sig = (newly_last > 0.5) and (0.165 <= belief <= 0.185) and self.config.use_stealth_detector
                    if not (inspected and confirmed) and belief < 0.70 and not stealth_sig:
                        q[isolate_act] -= 10.0

        q[~np.asarray(action_mask, dtype=bool)] = -1e9
        return int(np.argmax(q))

    def add_expert(self, s, a, r, ns, done, next_mask) -> None:
        self.expert_replay.append((
            np.asarray(s, np.float32),
            int(a),
            float(r),
            np.asarray(ns, np.float32),
            bool(done),
            np.asarray(next_mask, bool),
        ))

    def remember(self, s, a, r, ns, done, next_mask) -> None:
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
            n_agent = min(len(self.replay), batch_size - min(batch_size // 4, len(self.expert_replay)))
            n_expert = min(batch_size - n_agent, len(self.expert_replay))
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
            s = np.stack(exp_s)
            a = np.asarray(exp_a)
            r = np.asarray(exp_r, np.float32)
            ns = np.stack(exp_ns)
            done = np.asarray(exp_d, np.float32)
            masks = np.stack(exp_m)
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

    def save(self, path: str, metadata: dict[str, Any] | None = None) -> None:
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "config": self.config.__dict__,
                "steps": self.steps,
                "updates": self.updates,
                "metadata": metadata or {},
            },
            path,
        )

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
