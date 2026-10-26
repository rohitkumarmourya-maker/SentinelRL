#!/usr/bin/env python3
"""Verify genuine raw results/checkpoints; optional single-episode inference replay."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import source_digest
from src.environment import SentinelEnv

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--replay', action='store_true', help='Explicitly run one saved DQN episode and match its recorded raw row')
    args = parser.parse_args()
    output = args.output
    results = output / 'results'
    source_hash = source_digest()
    run_rows = []
    for algorithm, mode in (('ddqn', 'history'), ('ddqn', 'no_memory'), ('tabular_q', 'history')):
        for seed in (17, 29, 43):
            name = f'{algorithm}_{mode}_seed{seed}' if algorithm == 'ddqn' else f'{algorithm}_seed{seed}'
            manifest = json.loads((results / f'manifest_{name}.json').read_text())
            assert manifest['execution_status'] == 'COMPLETE'
            assert manifest['source_sha256'] == source_hash
            assert manifest['environment_steps'] == 120000
            extension = '.pt' if algorithm == 'ddqn' else '.json'
            checkpoint = output / 'models' / (name + extension)
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == manifest['checkpoint_sha256']
            episodes = read_csv(results / f'training_{name}.csv')
            updates = read_csv(results / f'updates_{name}.csv')
            assert len(episodes) == 2500 and [int(r['episode']) for r in episodes] == list(range(2500))
            assert len(updates) == manifest['updates']
            for row in episodes:
                assert np.isfinite(float(row['return']))
            if algorithm == 'ddqn':
                assert manifest['parameters_changed']
                assert manifest['initial_parameter_hash'] != manifest['final_parameter_hash']
                assert manifest['updates'] == 29751
                assert all(np.isfinite(float(row['loss'])) and np.isfinite(float(row['gradient_norm'])) for row in updates)
            else:
                assert manifest['updates'] == 120000
                assert all(np.isfinite(float(row['td_error'])) for row in updates)
            returns = [float(row['return']) for row in episodes]
            run_rows.append({'run': name, 'training_seed': seed, 'episodes': 2500, 'transitions': 120000,
                             'updates': manifest['updates'], 'elapsed_seconds': manifest['elapsed_seconds'],
                             'initial100_mean_training_return': float(np.mean(returns[:100])),
                             'last100_mean_training_return': float(np.mean(returns[-100:]))})
    action_rows = []
    for prefix, expected_rows in (('evaluation', 2100), ('sensitivity', 6300)):
        rows = read_csv(results / f'{prefix}_episodes.csv')
        assert len(rows) == expected_rows
        keys = [(r['policy'], r['training_seed'], r['scenario_index'], r['alert_sensitivity']) for r in rows]
        assert len(keys) == len(set(keys))
        assert all(sum(int(r[f'action_{a}']) for a in range(25)) == 48 for r in rows)
        assert all(int(r['invalid_actions']) == 0 and int(r['action_costs']) <= 6 for r in rows)
        assert all(np.isfinite(float(r['return'])) for r in rows)
        manifest = json.loads((results / f'{prefix}_manifest.json').read_text())
        assert manifest['source_sha256'] == source_hash and manifest['execution_status'] == 'COMPLETE'
        if prefix == 'evaluation':
            for policy in sorted({r['policy'] for r in rows}):
                selected = [r for r in rows if r['policy'] == policy]
                action_rows.append({'policy': policy, 'paired_episode_rows': len(selected),
                                    'wait': float(np.mean([int(r['action_0']) for r in selected])),
                                    **{kind: float(np.mean([sum(int(r[f'action_{a}']) for a in range(base, base + 6)) for r in selected]))
                                       for kind, base in (('inspect', 1), ('patch', 7), ('isolate', 13), ('decoy', 19))}})
    facts = {'source_sha256': source_hash, 'training_runs': run_rows,
             'total_training_transitions': sum(r['transitions'] for r in run_rows),
             'sum_logged_training_elapsed_seconds': sum(r['elapsed_seconds'] for r in run_rows),
             'total_ddqn_gradient_updates': sum(r['updates'] for r in run_rows if r['run'].startswith('ddqn')),
             'main_evaluation_rows': 2100, 'sensor_evaluation_rows': 6300,
             'seed_policy_scenario_blocks': '3 paired seeds × 7 policies × 100 scenarios; ×3 sensor conditions',
             'actions_mean_per_episode': action_rows,
             'scientific_status': 'Prespecified final checkpoints; no tuning/checkpoint selection on either evaluation set',
             'verification': 'Raw row/count/finite-value, checkpoint-hash, changed-parameter, source-hash and legality checks passed'}
    (results / 'RUN_FACTS.json').write_text(json.dumps(facts, indent=2) + '\n')
    with (results / 'action_type_means.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(action_rows[0])); writer.writeheader(); writer.writerows(action_rows)
    if args.replay:
        from src.dqn import DQNAgent
        policy = DQNAgent.load(output / 'models/ddqn_history_seed17.pt')
        assert policy.checkpoint_metadata['source_sha256'] == source_hash
        env = SentinelEnv(seed=300000)
        observation, info = env.reset(300000)
        for _ in range(48):
            action = policy.act(observation, info['action_mask'])
            observation, _, done, _, info = env.step(action)
            if done:
                break
        expected = next(r for r in read_csv(results / 'evaluation_episodes.csv') if r['policy'] == 'ddqn' and r['training_seed'] == '17' and r['scenario_index'] == '0')
        for key, value in env.metrics().items():
            assert np.isclose(value, float(expected[key]), rtol=0, atol=1e-9), (key, value, expected[key])
        assert all(env.action_counts[a] == int(expected[f'action_{a}']) for a in range(25))
        (results / 'replay_smoke.json').write_text(json.dumps({'status': 'PASSED', 'checkpoint': 'models/ddqn_history_seed17.pt',
                  'environment_seed': 300000, 'source_sha256': source_hash, 'raw_metric_and_action_counts_match': True,
                  'actual_metrics': env.metrics()}, indent=2) + '\n')
    print(json.dumps({'status': 'PASSED', 'training_runs': len(run_rows), 'raw_evaluation_rows': 8400,
                      'source_sha256': source_hash, 'training_elapsed_seconds_sum': facts['sum_logged_training_elapsed_seconds'],
                      'single_episode_replay': 'passed' if args.replay else 'not requested'}, indent=2))


if __name__ == '__main__':
    main()
