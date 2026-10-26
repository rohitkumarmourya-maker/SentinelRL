#!/usr/bin/env python3
"""Plot genuine recorded CSV evidence; fail when evaluation data is absent."""
import argparse
import csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    results = args.output / 'results'
    summary_path = results / 'evaluation_summary.csv'
    if not summary_path.exists():
        parser.error('No revised evaluation CSV exists. Run evaluation before requesting an empirical graph.')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    with summary_path.open() as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for ax, metric, label in zip(axes, ('return', 'critical_compromises'), ('Episode return (higher is better)', 'Critical hosts ever compromised (lower is better)')):
        selected = [r for r in rows if r['metric'] == metric and float(r['alert_sensitivity']) == .55]
        names = [r['policy'] for r in selected]
        y = np.array([float(r['mean']) for r in selected])
        low = np.maximum(0, y - np.array([float(r['ci95_low']) for r in selected]))
        high = np.maximum(0, np.array([float(r['ci95_high']) for r in selected]) - y)
        ax.barh(names, y, xerr=np.stack((low, high)), capsize=3, color=['#0b766e' if n == 'ddqn' else '#8194a8' for n in names])
        ax.set_xlabel(label)
    fig.suptitle('Revised held-out evaluation: crossed-bootstrap 95% intervals; 3 training seeds')
    fig.savefig(results / 'evaluation_comparison.png', dpi=200)
    plt.close(fig)
    paths = sorted(results.glob('training_ddqn_history_seed*.csv'))
    if paths:
        fig, ax = plt.subplots(figsize=(6.5, 3.5), constrained_layout=True)
        for path in paths:
            with path.open() as f:
                training = list(csv.DictReader(f))
            returns = np.array([float(r['return']) for r in training])
            window = min(50, len(returns))
            rolling = np.convolve(returns, np.ones(window) / window, mode='valid')
            ax.plot(np.arange(len(rolling)) + window, rolling, label=path.stem.split('_')[-1])
        ax.set(xlabel='Training episode', ylabel='Return (50-episode moving mean)', title='Actual epsilon-greedy training returns; not held-out performance')
        ax.legend(frameon=False)
        fig.savefig(results / 'training_curve.png', dpi=200)
        plt.close(fig)
    print(f'Figures written from genuine CSV rows in {results}')


if __name__ == '__main__':
    main()
