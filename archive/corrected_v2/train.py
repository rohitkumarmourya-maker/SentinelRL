#!/usr/bin/env python3
"""Training entry point. Requires explicit model-execution opt-in."""
import argparse
from pathlib import Path
from src.protocol import TRAINING_SEEDS, TRAINING_EPISODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-model-run', action='store_true', help='Explicitly opt in to real model training')
    parser.add_argument('--algorithm', choices=('ddqn', 'tabular_q', 'all'), default='all')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(TRAINING_SEEDS))
    parser.add_argument('--episodes', type=int, default=TRAINING_EPISODES)
    parser.add_argument('--observation-mode', choices=('history', 'no_memory'), default='history')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if not args.execute_model_run:
        parser.error('Add --execute-model-run to explicitly request real model training.')
    if args.episodes < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('episodes must be positive; seeds must be unique nonnegative integers')
    from src.experiment import train
    train(args.algorithm, args.seeds, args.episodes, args.observation_mode, args.output)


if __name__ == '__main__':
    main()
