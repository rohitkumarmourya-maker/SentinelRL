#!/usr/bin/env python3
"""Safe default: show only public telemetry under a scripted environment exercise."""
import argparse
import json
from pathlib import Path
from src.environment import SentinelEnv, INSPECT_BASE, ISOLATE_BASE, PATCH_BASE, DECOY_BASE, WAIT, action_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--checkpoint', type=Path, help='Optional revised DQN checkpoint; requires model-run opt-in')
    parser.add_argument('--execute-model-run', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    if args.checkpoint and not args.execute_model_run:
        parser.error('Checkpoint inference needs explicit --execute-model-run')
    agent = None
    if args.checkpoint:
        from src.dqn import DQNAgent
        agent = DQNAgent.load(args.checkpoint)
    env = SentinelEnv(seed=args.seed)
    observation, info = env.reset(args.seed)
    # Fixed API exercise. This is not an RL policy, evaluation or performance claim.
    exercise = [INSPECT_BASE, PATCH_BASE + 3, DECOY_BASE + 5, INSPECT_BASE + 4, ISOLATE_BASE, WAIT]
    print('Checkpoint inference' if agent else 'Environment API exercise only; no learned policy has been run.')
    print(json.dumps(env.public_snapshot(), indent=2))
    for step in range(min(args.steps, env.config.horizon)):
        action = agent.act(observation, info['action_mask']) if agent else exercise[step] if step < len(exercise) else WAIT
        if not info['action_mask'][action]:
            action = WAIT
        observation, _, terminated, _, info = env.step(action)
        print(f't={env.t}: {action_name(action)}; budget={env.budget}; alarms={env.alert.astype(int).tolist()}')
        if terminated:
            break
    print('No performance score is reported for this API exercise.')


if __name__ == '__main__':
    main()
