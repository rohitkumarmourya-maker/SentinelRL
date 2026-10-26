#!/usr/bin/env python3
"""Print the complete future protocol; run it only with explicit model opt-in."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-model-run', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    suffix = ['--output', str(args.output)]
    commands = [
        [sys.executable, 'train.py', '--execute-model-run', '--algorithm', 'all', *suffix],
        [sys.executable, 'train.py', '--execute-model-run', '--algorithm', 'ddqn', '--observation-mode', 'no_memory', *suffix],
        [sys.executable, 'evaluate.py', '--execute-model-run', '--include-no-memory', *suffix],
        [sys.executable, 'evaluate.py', '--execute-model-run', '--include-no-memory', '--sensitivity', *suffix],
        [sys.executable, 'scripts/plot_results.py', '--output', str(args.output)],
    ]
    if not args.execute_model_run:
        print('DRY RUN: no models will execute. Future commands:')
    for command in commands:
        print(' '.join(command), flush=True)
        if args.execute_model_run:
            subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
