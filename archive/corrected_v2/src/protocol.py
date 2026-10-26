"""Frozen future experiment protocol. Defining this file does not execute models."""
from dataclasses import asdict
from .environment import EnvConfig

TRAINING_SEEDS = (17, 29, 43)
TRAINING_EPISODES = 2500
EVALUATION_EPISODES = 100
SENSITIVITIES = (.35, .55, .75)


def training_seed(seed, episode):
    return 1000000 * seed + episode


def evaluation_seed(episode):
    return 300000 + 97 * episode


def sensitivity_seed(episode):
    return 400000 + 97 * episode


def protocol_dict():
    return {"version": 2, "status": "FROZEN_FUTURE_PROTOCOL",
            "training_seeds": list(TRAINING_SEEDS), "training_episodes_per_seed": TRAINING_EPISODES,
            "evaluation_episodes_per_seed_policy": EVALUATION_EPISODES,
            "training_episode_seed": "1000000*training_seed + episode_index",
            "heldout_episode_seed": "300000 + 97*episode_index",
            "sensitivity_episode_seed": "400000 + 97*episode_index (same scenarios across sensor conditions)",
            "sensor_sensitivities": list(SENSITIVITIES), "checkpoint_selection": "final fixed training budget, no test selection",
            "environment": asdict(EnvConfig()),
            "ablations": ["untrained_initial_network", "retrained_no_memory_ddqn"],
            "baselines": ["tabular_q", "alert_rule_diagnostic", "random_legal", "no_op"],
            "uncertainty": "95% crossed bootstrap resampling training-seed rows and paired-scenario columns; 3 seeds is a small sample"}
