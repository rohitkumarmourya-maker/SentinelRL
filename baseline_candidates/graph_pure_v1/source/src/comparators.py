"""Lawful comparators on the unchanged corrected POMDP.

Historical model/source identity is checked against its own immutable snapshot;
adding new comparator modules does not waive or silently replace old validation.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch
from .belief import ExactBelief
from .environment import EnvConfig, N_HOSTS, FEATURES_PER_HOST, INSPECT_BASE, PATCH_BASE

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'archive/corrected_v2'
ORIGINAL = ROOT / 'vendor/original_v1'
PREVIOUS_SOURCE_HASH = '342709f0edec919c70b44d35d814ab56161f126926482ef0b35e21171220c40e'
PREVIOUS_SEEDS = (17, 29, 43)


def source_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob('*.py')):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_previous_source():
    if source_hash(PREVIOUS / 'src') != PREVIOUS_SOURCE_HASH:
        raise ValueError('previous source snapshot hash mismatch')
    for name in ('environment.py','dqn.py','policies.py'):
        if (ROOT/'src'/name).read_bytes() != (PREVIOUS/'src'/name).read_bytes():
            raise ValueError(f'previous runtime module differs from trained snapshot: {name}')


def verify_original_source():
    # Compare every preserved historical Python file to the original ZIP snapshot.
    for path in ORIGINAL.glob('*.py'):
        expected = ROOT / 'legacy/source/src' / path.name
        if path.read_bytes() != expected.read_bytes():
            raise ValueError(f'original comparison source modified: {path.name}')


class PreviousPolicy:
    def __init__(self, kind: str, seed: int):
        verify_previous_source()
        if kind == 'previous_tabular':
            from .policies import TabularQAgent
            self.agent = TabularQAgent.load(PREVIOUS/'models'/f'tabular_q_seed{seed}.json')
        else:
            from .dqn import DQNAgent
            mode = 'history' if kind == 'previous_ddqn' else 'no_memory'
            self.agent = DQNAgent.load(PREVIOUS/'models'/f'ddqn_{mode}_seed{seed}.pt')
        metadata = self.agent.checkpoint_metadata
        if metadata.get('source_sha256') != PREVIOUS_SOURCE_HASH:
            raise ValueError('previous checkpoint training-source mismatch')
        from dataclasses import asdict
        if metadata.get('protocol', {}).get('environment') != asdict(EnvConfig()):
            raise ValueError('previous checkpoint environment mismatch')

    def reset(self, observation, mask=None):
        pass

    def act(self, observation, mask, explore=False):
        return self.agent.act(observation, mask, explore=False)


class Lawful58Adapter:
    """Causal original-format features from public observations/actions only.

    The original privileged new-compromise channel is unavailable and fixed0.
    True-alert-derived detected count is replaced by explicit positive diagnoses.
    Inspection history never resets on an unseen infection.
    """
    def __init__(self, config=None):
        self.filter = ExactBelief(config=config)
        self.previous_action = None
        self.initialized = False

    def reset(self, observation):
        self.filter.reset(observation)
        self.previous_action = None
        self.initialized = True
        return self.vector(observation)

    def advance(self, observation):
        if not self.initialized:
            return self.reset(observation)
        if self.previous_action is not None:
            self.filter.update(self.previous_action, observation)
            self.previous_action = None
        return self.vector(observation)

    def vector(self, observation):
        public = np.asarray(observation, np.float32)
        nodes = public[:54].reshape(6, 9)
        inspected = nodes[:, 2] > 0
        confirmed = nodes[:, 2] == 1
        old_nodes = np.column_stack((self.filter.marginals, nodes[:, 0], inspected, confirmed,
                                     nodes[:, 4], nodes[:, 5], nodes[:, 6], nodes[:, 7], nodes[:, 8]))
        globals_ = [public[-2], public[-1], float(confirmed.sum()) / 6., 0.]
        vector = np.concatenate((old_nodes.reshape(-1), globals_)).astype(np.float32)
        assert vector.shape == (58,)
        return vector

    @staticmethod
    def legacy_mask(observation, true_mask):
        """Restore original inspect affordance using public history, never hidden state."""
        mask = np.array(true_mask, bool, copy=True)
        nodes = np.asarray(observation)[:54].reshape(6, 9)
        mask[INSPECT_BASE:PATCH_BASE] &= (nodes[:, 2] == 0) | (nodes[:, 0] > .5)
        return mask


class OriginalPolicy:
    def __init__(self, kind: str, seed: int = 0, config=None):
        verify_original_source()
        from vendor.original_v1 import policies as original_policies
        self.adapter = Lawful58Adapter(config)
        self.kind = kind
        if kind in ('original_full_transfer','original_no_stealth_transfer'):
            from vendor.original_v1.dqn import DQNAgent
            path = ORIGINAL/'models'/f'sentinel_dqn_seed{seed}.pt'
            expected = ROOT/'legacy/models'/path.name
            if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(expected.read_bytes()).digest():
                raise ValueError('original checkpoint hash mismatch')
            self.agent = DQNAgent.load(str(path), device='cpu')
            if kind == 'original_no_stealth_transfer':
                self.agent.config.use_stealth_detector = False
                self.agent.online.use_stealth_detector = False
            self.is_dqn = True
        else:
            factory = {'original_alert': original_policies.AlertResponderPolicy,
                       'original_threshold': original_policies.ThresholdDefensePolicy,
                       'original_patch': original_policies.PatchPriorityPolicy,
                       'original_random': lambda: original_policies.RandomLegalPolicy(seed)}[kind]
            self.agent = factory()
            self.is_dqn = False

    def reset(self, observation, mask=None):
        self.adapter.reset(observation)

    def act(self, observation, mask, explore=False):
        adapted = self.adapter.advance(observation)
        # Random legal remains uniform over the actual corrected action set.
        legacy_mask = mask if self.kind == 'original_random' else self.adapter.legacy_mask(observation, mask)
        action = self.agent.act(adapted, legacy_mask, explore=False) if self.is_dqn else self.agent.act(adapted, legacy_mask)
        self.adapter.previous_action = action
        return int(action)


PRETRAINED_POLICIES = ('previous_ddqn','previous_no_memory','previous_tabular',
                      'original_alert','original_threshold','original_patch','original_random',
                      'original_full_transfer','original_no_stealth_transfer')


def make_comparator(kind, model_seed=17, scenario_seed=0, config=None):
    torch.set_num_threads(1)
    if kind.startswith('previous_'):
        return PreviousPolicy(kind, model_seed)
    if kind.startswith('original_'):
        seed = scenario_seed if kind == 'original_random' else model_seed
        return OriginalPolicy(kind, seed, config)
    raise ValueError(f'unknown comparator {kind}')


def comparator_seeds(kind):
    return PREVIOUS_SEEDS if kind in ('previous_ddqn','previous_no_memory','previous_tabular','original_full_transfer','original_no_stealth_transfer') else (0,)


class BeliefThresholdPolicy:
    """Validation-tuned diagnostic comparator; never used inside a learned policy."""
    def __init__(self, isolation_threshold=.8, inspection_threshold=.3, protection=False):
        self.isolation_threshold = isolation_threshold
        self.inspection_threshold = inspection_threshold
        self.protection = protection
        self.adapter = Lawful58Adapter()

    def reset(self, observation, mask=None):
        self.adapter.reset(observation)

    def act(self, observation, mask, explore=False):
        adapted = self.adapter.advance(observation)
        nodes = np.asarray(observation)[:54].reshape(6,9)
        marginal = adapted[:54].reshape(6,9)[:,0]
        criticality = nodes[:,5]
        # Identical known graph available to every controller.
        from .environment import ADJ, CRITICALITY, ISOLATE_BASE, DECOY_BASE
        source_importance = .5 + ADJ @ CRITICALITY
        eligible = mask[ISOLATE_BASE:DECOY_BASE] & (marginal >= self.isolation_threshold)
        if eligible.any():
            action = ISOLATE_BASE + int(np.argmax(np.where(eligible, marginal*source_importance,-1.)))
        else:
            remaining = round(float(observation[-2])*6)
            inspect = mask[INSPECT_BASE:PATCH_BASE] & (marginal >= self.inspection_threshold)
            inspect &= (nodes[:,2] != 1) & ((nodes[:,2] == 0) | (nodes[:,3] >= 2/48))
            if remaining >= 2 and inspect.any():
                host = int(np.argmax(np.where(inspect,marginal*(1-marginal)*source_importance,-1.)))
                action = INSPECT_BASE+host
            elif self.protection and remaining >= 3:
                exposure = ADJ.T @ (marginal*(1-nodes[:,7]))
                score = exposure*(1-marginal)*nodes[:,4]*(1+criticality)
                patch = mask[PATCH_BASE:ISOLATE_BASE]
                action = PATCH_BASE+int(np.argmax(np.where(patch,score,-1.))) if patch.any() and np.max(np.where(patch,score,0))>.01 else 0
            else:
                action=0
        self.adapter.previous_action=int(action)
        return int(action)
