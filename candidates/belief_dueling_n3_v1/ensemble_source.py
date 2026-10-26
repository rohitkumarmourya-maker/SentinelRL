"""Exact-belief dueling multi-step Double DQN, trained by PyTorch autograd.

The selected action is the argmax learned Q among legal actions. There are no
teacher actions, hand-written action preferences, hidden-state features, or
post-processing shields. A forced-wait suffix is folded into its preceding
transition using the actual environment rewards, preserving the return exactly.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn
from .belief import ExactBelief, FEATURE_DIM
from .environment import ADJ, EnvConfig, N_ACTIONS


@dataclass(frozen=True)
class ChampionConfig:
    gamma: float = 1.
    representation: str = "raw"
    architecture: str = "dueling_mlp"
    learning_rate: float = 5e-4
    batch_size: int = 128
    replay_capacity: int = 100000
    learning_starts: int = 1000
    train_every: int = 1
    target_sync_updates: int = 500
    epsilon_start: float = 1.
    epsilon_end: float = .02
    epsilon_decay_steps: int = 300000
    hidden_size: int = 256
    n_step: int = 3
    grad_clip: float = 10.
    huber_beta: float = 5.
    td_loss: str = "huber"
    prioritized_replay: bool = False
    priority_alpha: float = .6
    importance_beta: float = .4
    importance_anneal_updates: int = 200000
    compress_forced_wait: bool = True


class DuelingQNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        self.encoder = nn.Sequential(nn.Linear(FEATURE_DIM, h), nn.ReLU(),
                                     nn.Linear(h, h), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 1))
        self.advantage = nn.Sequential(nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, N_ACTIONS))

    def forward(self, x):
        encoded = self.encoder(x)
        advantage = self.advantage(encoded)
        return self.value(encoded) + advantage - advantage.mean(dim=-1, keepdim=True)


class SharedHostQNetwork(nn.Module):
    """Shared learned host/action values with public graph identity and joint belief."""
    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        self.register_buffer("topology", torch.from_numpy(ADJ.astype(np.float32)))
        self.joint = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.host = nn.Sequential(nn.Linear(9 + 1 + 2 + 6 + 32, h), nn.ReLU(),
                                  nn.Linear(h, h), nn.ReLU())
        self.actions = nn.Linear(h, 4)
        self.wait = nn.Sequential(nn.Linear(h + 34, h), nn.ReLU(), nn.Linear(h, 1))
        self.value = nn.Sequential(nn.Linear(h + 34, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        batch = x.shape[0]
        nodes = x[:, :54].reshape(batch, 6, 9)
        globals_ = x[:, 54:56]
        joint = self.joint(x[:, 56:120])
        marginal = x[:, 120:126, None]
        inputs = torch.cat((nodes, marginal, globals_[:,None,:].expand(-1,6,-1),
                            self.topology[None].expand(batch,-1,-1),
                            joint[:,None,:].expand(-1,6,-1)), dim=2)
        host = self.host(inputs)
        global_features = torch.cat((host.mean(1), globals_, joint), dim=1)
        advantages = torch.cat((self.wait(global_features),
                                self.actions(host).transpose(1,2).reshape(batch,24)), dim=1)
        return self.value(global_features) + advantages - advantages.mean(1, keepdim=True)


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


class PrioritizedReplay(Replay):
    """Proportional TD-error replay with an exact segment tree and IS correction."""
    def __init__(self, capacity, seed, config):
        super().__init__(capacity, seed)
        self.config = config
        self.leaf_count = 1 << (capacity - 1).bit_length()
        self.tree = np.zeros(2 * self.leaf_count, dtype=np.float64)
        self.max_priority = 1.
        self.samples = 0

    def _set(self, indices, priorities):
        indices, inverse = np.unique(np.asarray(indices, dtype=np.int64), return_inverse=True)
        merged = np.zeros(len(indices), dtype=np.float64)
        np.maximum.at(merged, inverse, priorities)
        leaves = indices + self.leaf_count
        self.tree[leaves] = merged ** self.config.priority_alpha
        parents = np.unique(leaves // 2)
        while len(parents) and parents[0] > 0:
            self.tree[parents] = self.tree[2 * parents] + self.tree[2 * parents + 1]
            parents = np.unique(parents // 2)

    def add(self, s, a, r, ns, discount, mask):
        index = self.position
        super().add(s, a, r, ns, discount, mask)
        self._set(np.array([index]), np.array([self.max_priority]))

    def sample(self, n):
        total = self.tree[1]
        masses = (np.arange(n) + self.rng.random(n)) * total / n
        nodes = np.ones(n, dtype=np.int64)
        while nodes[0] < self.leaf_count:
            left = 2 * nodes
            right = masses >= self.tree[left]
            masses -= np.where(right, self.tree[left], 0.)
            nodes = left + right
        indices = nodes - self.leaf_count
        probability = self.tree[nodes] / total
        beta = min(1., self.config.importance_beta + (1 - self.config.importance_beta) * self.samples / self.config.importance_anneal_updates)
        weights = (self.size * probability) ** (-beta)
        weights /= weights.max()
        self.samples += 1
        batch = tuple(torch.from_numpy(x[indices]) for x in (self.s, self.a, self.r, self.ns, self.discount, self.mask))
        return (*batch, torch.from_numpy(weights.astype(np.float32)), indices)

    def update_priorities(self, indices, errors):
        priorities = np.abs(errors) + 1e-3
        self.max_priority = max(self.max_priority, float(priorities.max()))
        self._set(indices, priorities)


class ChampionAgent:
    def __init__(self, seed=0, config: ChampionConfig | None = None, env_config: EnvConfig | None = None):
        self.seed, self.config = seed, config or ChampionConfig()
        self.env_config = env_config or EnvConfig()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        network = {"dueling_mlp": DuelingQNetwork, "shared_host": SharedHostQNetwork}[self.config.architecture]
        self.online = network(self.config)
        self.target = network(self.config)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.replay = (PrioritizedReplay(self.config.replay_capacity, seed+1, self.config)
                       if self.config.prioritized_replay else Replay(self.config.replay_capacity, seed+1))
        self.environment_steps = self.updates = self.decision_transitions = 0
        self.initial_parameter_hash = self.parameter_hash()
        self.belief = ExactBelief(self.env_config)
        self.current_features = self.last_action = self.last_time = None
        self.trajectory = []
        self.checkpoint_metadata = {}

    def _represent(self, features):
        if self.config.representation == "sufficient":
            features = features.copy()
            nodes = features[:54].reshape(6,9)
            confirmed = (nodes[:,2] == 1.).copy()
            nodes[:,:4] = 0.
            nodes[:,2] = confirmed
        elif self.config.representation != "raw":
            raise ValueError("unknown feature representation")
        return features

    def reset(self, observation, mask=None):
        self.current_features = self._represent(self.belief.reset(observation))
        self.last_action, self.last_time = None, 0
        self.trajectory = []
        return self.current_features

    def _prepare(self, observation):
        current_time = round(float(observation[-1]) * self.env_config.horizon)
        if self.last_time is None or current_time == 0 and self.last_time != 0:
            self.reset(observation)
        elif current_time != self.last_time:
            self.current_features = self._represent(self.belief.update(self.last_action, observation))
            self.last_time = current_time
        return self.current_features

    def epsilon(self):
        c = self.config
        return c.epsilon_start + min(1., self.environment_steps/c.epsilon_decay_steps) * (c.epsilon_end-c.epsilon_start)

    def act(self, observation, mask, explore=False):
        features = self._prepare(observation)
        legal = np.flatnonzero(mask)
        if len(legal) == 1:
            action = int(legal[0])
        elif explore and self.rng.random() < self.epsilon():
            action = int(self.rng.choice(legal))
        else:
            with torch.no_grad():
                q = self.online(torch.from_numpy(features)[None]).numpy()[0]
            action = int(np.argmax(np.where(mask, q, -np.inf)))
        self.acted_features = features.copy()
        self.acted_mask = np.array(mask, copy=True)
        self.last_action = action
        return action

    def observe(self, state, action, reward, next_state, done, next_mask):
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
                batch = self.replay.sample(c.batch_size)
                states, actions, rewards, next_states, discounts, masks = batch[:6]
                weights = batch[6] if c.prioritized_replay else torch.ones(c.batch_size)
                predictions = self.online(states).gather(1, actions[:,None]).squeeze(1)
                with torch.no_grad():
                    choices = self.online(next_states).masked_fill(~masks, -torch.inf).argmax(1, keepdim=True)
                    values = self.target(next_states).gather(1, choices).squeeze(1)
                    targets = rewards + discounts * values
                if c.td_loss == "mse":
                    per_item_loss = nn.functional.mse_loss(predictions, targets, reduction="none")
                elif c.td_loss == "huber":
                    per_item_loss = nn.functional.smooth_l1_loss(predictions, targets, beta=c.huber_beta, reduction="none")
                else:
                    raise ValueError("unknown temporal-difference loss")
                loss = (weights * per_item_loss).mean()
                if c.prioritized_replay:
                    self.replay.update_priorities(batch[7], (targets - predictions).detach().numpy())
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
        torch.save({"format_version": 3, "algorithm": "exact_belief_dueling_nstep_double_dqn",
                    "feature_dim": FEATURE_DIM, "seed": self.seed, "config": asdict(self.config),
                    "environment": asdict(self.env_config), "online": self.online.state_dict(),
                    "target": self.target.state_dict(), "environment_steps": self.environment_steps,
                    "updates": self.updates, "decision_transitions": self.decision_transitions,
                    "initial_parameter_hash": self.initial_parameter_hash,
                    "parameter_hash": self.parameter_hash(), "metadata": metadata or {}}, path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 3 or payload.get("feature_dim") != FEATURE_DIM:
            raise ValueError("incompatible champion checkpoint")
        agent = cls(payload['seed'], ChampionConfig(**payload['config']), EnvConfig(**payload['environment']))
        agent.online.load_state_dict(payload['online']); agent.target.load_state_dict(payload['target'])
        for name in ('environment_steps', 'updates', 'decision_transitions', 'initial_parameter_hash'):
            setattr(agent, name, payload[name])
        agent.checkpoint_metadata = payload['metadata']
        if agent.parameter_hash() != payload['parameter_hash']:
            raise ValueError("checkpoint parameter hash mismatch")
        agent.online.eval()
        return agent


class EnsembleChampionAgent:
    """Arithmetic mean of learned Q functions with a single shared causal belief.

    Members must use the same frozen representation, architecture and discount.
    This is one deployment policy; its members are not independent ensembles.
    """
    def __init__(self, agents):
        if not agents:
            raise ValueError("at least one learned member is required")
        self.members = list(agents)
        reference = self.members[0]
        for agent in self.members:
            if (agent.config.representation != reference.config.representation or
                agent.config.architecture != reference.config.architecture or
                agent.config.gamma != reference.config.gamma or
                agent.env_config != reference.env_config):
                raise ValueError("Q averaging requires a common representation and task")
            agent.online.eval()
        self.belief_owner = reference

    def reset(self, observation, mask=None):
        return self.belief_owner.reset(observation, mask)

    def act(self, observation, mask, explore=False):
        if explore:
            raise ValueError("the frozen Q ensemble is an evaluation policy")
        features = self.belief_owner._prepare(observation)
        with torch.no_grad():
            tensor = torch.from_numpy(features)[None]
            q = torch.stack([member.online(tensor) for member in self.members]).mean(0).numpy()[0]
        action = int(np.argmax(np.where(mask, q, -np.inf)))
        self.belief_owner.last_action = action
        return action

    @classmethod
    def from_checkpoints(cls, paths):
        return cls([ChampionAgent.load(path) for path in paths])
