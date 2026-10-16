"""Simple, explicit evaluation baselines (not the primary solver)."""
from __future__ import annotations
import numpy as np
from .environment import (WAIT, N_HOSTS, INSPECT_BASE, PATCH_BASE, ISOLATE_BASE,
                          DECOY_BASE, FEATURES_PER_HOST, VULNERABILITY, CRITICALITY)

class RandomLegalPolicy:
    name = "Random legal"
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def act(self, obs, mask): return int(self.rng.choice(np.flatnonzero(mask)))

class PatchPriorityPolicy:
    """Harden critical assets first; isolate only when telemetry is very strong."""
    name = "Patch priority"
    def act(self, obs, mask):
        x = obs[:N_HOSTS * FEATURES_PER_HOST].reshape(N_HOSTS, FEATURES_PER_HOST)
        # If an alert is strong and host is not yet inspected, inspect before isolating.
        suspicious = [i for i in range(N_HOSTS) if mask[INSPECT_BASE+i] and x[i, 0] > 0.55]
        if suspicious:
            i = max(suspicious, key=lambda j: x[j, 0] * CRITICALITY[j])
            return INSPECT_BASE + i
        confirmed = [i for i in range(N_HOSTS) if x[i, 2] and x[i, 3] and mask[ISOLATE_BASE+i]]
        if confirmed:
            return ISOLATE_BASE + max(confirmed, key=lambda j: CRITICALITY[j])
        patchable = [i for i in range(N_HOSTS) if mask[PATCH_BASE+i]]
        if patchable:
            i = max(patchable, key=lambda j: CRITICALITY[j] * VULNERABILITY[j])
            return PATCH_BASE + i
        return WAIT

class AlertResponderPolicy:
    """Inspect the riskiest unverified host, isolate confirmed compromise, then harden."""
    name = "Alert responder"
    def act(self, obs, mask):
        x = obs[:N_HOSTS * FEATURES_PER_HOST].reshape(N_HOSTS, FEATURES_PER_HOST)
        confirmed = [i for i in range(N_HOSTS) if x[i, 2] > 0.5 and x[i, 3] > 0.5 and not x[i, 7] and mask[ISOLATE_BASE+i]]
        if confirmed:
            i = max(confirmed, key=lambda j: CRITICALITY[j])
            return ISOLATE_BASE + i
        inspectable = [i for i in range(N_HOSTS) if mask[INSPECT_BASE+i] and (x[i, 1] > 0.5 or x[i, 0] > 0.27)]
        if inspectable:
            i = max(inspectable, key=lambda j: x[j, 0] * (0.5 + CRITICALITY[j]))
            return INSPECT_BASE + i
        # If telemetry is quiet, place a decoy near a high-value, vulnerable part of the graph.
        decoyable = [i for i in range(N_HOSTS) if mask[DECOY_BASE+i]]
        if decoyable:
            i = max(decoyable, key=lambda j: CRITICALITY[j] * VULNERABILITY[j])
            return DECOY_BASE + i
        patchable = [i for i in range(N_HOSTS) if mask[PATCH_BASE+i]]
        if patchable:
            i = max(patchable, key=lambda j: CRITICALITY[j] * VULNERABILITY[j])
            return PATCH_BASE + i
        return WAIT

class ThresholdDefensePolicy:
    """Threshold policy: quarantine only above a high posterior-risk threshold."""
    name = "Threshold defense"
    def act(self, obs, mask):
        x = obs[:N_HOSTS * FEATURES_PER_HOST].reshape(N_HOSTS, FEATURES_PER_HOST)
        likely = [i for i in range(N_HOSTS) if x[i, 0] >= 0.78 and not x[i, 7] and mask[ISOLATE_BASE+i]]
        if likely:
            return ISOLATE_BASE + max(likely, key=lambda j: CRITICALITY[j] * x[j,0])
        suspects = [i for i in range(N_HOSTS) if mask[INSPECT_BASE+i] and (x[i,1] > 0.5 or x[i,0] > 0.4)]
        if suspects:
            return INSPECT_BASE + max(suspects, key=lambda j: x[j,0])
        patchable = [i for i in range(N_HOSTS) if mask[PATCH_BASE+i]]
        if patchable:
            i = max(patchable, key=lambda j: CRITICALITY[j] * VULNERABILITY[j])
            return PATCH_BASE + i
        return WAIT
