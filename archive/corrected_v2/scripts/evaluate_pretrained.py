#!/usr/bin/env python3
"""Evaluate bundled checkpoints in a fresh directory without overwriting evidence."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-model-run', action='store_true')
    parser.add_argument('--output', type=Path, required=True, help='New directory; must not already exist')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 29, 43], choices=[17, 29, 43])
    parser.add_argument('--sensitivity', action='store_true')
    args = parser.parse_args()
    if not args.execute_model_run:
        parser.error('Add --execute-model-run to request real checkpoint inference/evaluation.')
    if args.output.exists():
        parser.error('Output already exists; choose a fresh directory to preserve evidence.')
    if args.episodes < 1 or len(args.seeds) != len(set(args.seeds)):
        parser.error('episodes must be positive and seeds unique')
    shutil.copytree(ROOT / 'models', args.output / 'models')
    command = [sys.executable, str(ROOT / 'evaluate.py'), '--execute-model-run', '--include-no-memory',
               '--output', str(args.output.resolve()), '--episodes', str(args.episodes),
               '--seeds', *map(str, args.seeds)]
    if args.sensitivity:
        command.append('--sensitivity')
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
