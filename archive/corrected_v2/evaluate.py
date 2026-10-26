#!/usr/bin/env python3
"""Held-out evaluation. Never substitutes old checkpoints or results."""
import argparse
from pathlib import Path
from src.protocol import TRAINING_SEEDS, EVALUATION_EPISODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-model-run', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(TRAINING_SEEDS))
    parser.add_argument('--episodes', type=int, default=EVALUATION_EPISODES)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--include-no-memory', action='store_true', help='Require separately retrained no-memory checkpoints')
    parser.add_argument('--sensitivity', action='store_true', help='Use disjoint sensor-sensitivity scenarios')
    args = parser.parse_args()
    if not args.execute_model_run:
        parser.error('Add --execute-model-run to explicitly request checkpoint evaluation.')
    if args.episodes < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('episodes must be positive; seeds must be unique nonnegative integers')
    from src.experiment import evaluate
    evaluate(args.output, args.seeds, args.episodes, args.include_no_memory, args.sensitivity)


if __name__ == '__main__':
    main()
