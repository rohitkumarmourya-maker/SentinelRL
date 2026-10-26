"""Explicit opt-in training/evaluation, with raw logs and provenance manifests."""
from __future__ import annotations
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import platform
import subprocess
import time
import numpy as np
from .environment import EnvConfig, SentinelEnv, N_ACTIONS, action_name
from .policies import AlertResponderPolicy, NoOpPolicy, RandomPolicy, TabularQAgent
from .protocol import (TRAINING_SEEDS, TRAINING_EPISODES, EVALUATION_EPISODES, SENSITIVITIES,
                       training_seed, evaluation_seed, sensitivity_seed, protocol_dict)
from .statistics import crossed_bootstrap

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("return", "total_compromises", "critical_compromises", "uncontained_at_end", "false_isolations", "successful_isolations", "action_costs", "invalid_actions", "confirmed_fraction", "censored_confirmation_delay")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("refusing to write empty evidence table")
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + '\n')


def source_digest():
    digest = hashlib.sha256()
    for path in sorted((ROOT / 'src').glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def metadata():
    import torch  # only reached by explicitly requested model-running commands
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "numpy": np.__version__, "torch": str(torch.__version__), "device": "cpu", "source_sha256": source_digest(),
            "git_original_head": subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() if (ROOT / '.git').exists() else 'see provenance/original-history.bundle',
            "protocol": protocol_dict()}


def train(algorithm='all', seeds=TRAINING_SEEDS, episodes=TRAINING_EPISODES, observation_mode='history', output=ROOT):
    from .dqn import DQNAgent, DQNConfig
    output = Path(output)
    run_metadata = metadata()
    run_metadata['execution_status'] = 'RUNNING'
    run_metadata['actual_episodes_per_seed'] = episodes
    run_metadata['actual_seeds'] = list(seeds)
    algorithms = ('ddqn', 'tabular_q') if algorithm == 'all' else (algorithm,)
    # Preflight every artifact before initializing a model or writing any file.
    intended = []
    for name in algorithms:
        for seed in seeds:
            run_name = f'{name}_{observation_mode}_seed{seed}' if name == 'ddqn' else f'{name}_seed{seed}'
            ext = '.pt' if name == 'ddqn' else '.json'
            intended.extend([output / 'models' / (run_name + ext),
                             output / 'results' / f'training_{run_name}.csv',
                             output / 'results' / f'updates_{run_name}.csv',
                             output / 'results' / f'manifest_{run_name}.json'])
            if name == 'ddqn':
                intended.append(output / 'models' / (run_name + '_initial.pt'))
    status_name = f'training_status_{algorithm}_{observation_mode}_' + '-'.join(map(str, seeds)) + '.json'
    intended.append(output / 'results' / status_name)
    existing = [str(path) for path in intended if path.exists()]
    if existing:
        raise FileExistsError('Preserving existing run artifacts. Choose a fresh --output directory or a distinct run: ' + ', '.join(existing))
    for name in algorithms:
        for seed in seeds:
            run_name = f'{name}_{observation_mode}_seed{seed}' if name == 'ddqn' else f'{name}_seed{seed}'
            checkpoint = output / 'models' / (run_name + ('.pt' if name == 'ddqn' else '.json'))
            if checkpoint.exists():
                raise FileExistsError(f'{checkpoint} exists; use a new --output directory to preserve evidence')
            agent = DQNAgent(seed, observation_mode=observation_mode) if name == 'ddqn' else TabularQAgent(seed)
            meta = {**run_metadata, 'algorithm': name, 'observation_mode': observation_mode}
            if name == 'ddqn':
                agent.save(checkpoint.with_name(checkpoint.stem + '_initial.pt'), {**meta, 'checkpoint': 'untrained_initial'})
            episode_rows, update_rows = [], []
            start = time.monotonic()
            log_path = output / 'results' / f'training_{run_name}.csv'
            updates_path = output / 'results' / f'updates_{run_name}.csv'
            write_json(output / 'results' / f'manifest_{run_name}.json', meta)
            for episode in range(episodes):
                env = SentinelEnv(seed=training_seed(seed, episode))
                observation, info = env.reset(training_seed(seed, episode))
                losses = []
                for _ in range(env.config.horizon):
                    action = agent.act(observation, info['action_mask'], explore=True)
                    next_observation, reward, terminated, truncated, next_info = env.step(action)
                    update = agent.observe(observation, action, reward, next_observation, terminated or truncated, next_info['action_mask'])
                    if update is not None:
                        update_rows.append({'episode': episode, 'environment_step': agent.environment_steps, **update})
                        losses.append(update.get('loss', update.get('td_error', 0.)))
                    observation, info = next_observation, next_info
                    if terminated or truncated:
                        break
                episode_rows.append({'algorithm': name, 'observation_mode': observation_mode, 'training_seed': seed,
                                     'episode': episode, 'environment_seed': env.seed, **env.metrics(),
                                     'epsilon': agent.epsilon(), 'environment_steps': agent.environment_steps,
                                     'gradient_updates' if name == 'ddqn' else 'q_updates': agent.updates,
                                     'mean_training_loss' if name == 'ddqn' else 'mean_td_error': float(np.mean(losses)) if losses else '',
                                     'elapsed_seconds': time.monotonic() - start})
                if (episode + 1) % 100 == 0 or episode + 1 == episodes:
                    write_csv(log_path, episode_rows)
                    if update_rows:
                        write_csv(updates_path, update_rows)
                    print(f'{run_name}: {episode + 1}/{episodes} episodes; updates={agent.updates}', flush=True)
            meta['execution_status'] = 'COMPLETE'
            meta['elapsed_seconds'] = time.monotonic() - start
            meta['environment_steps'], meta['updates'] = agent.environment_steps, agent.updates
            if name == 'ddqn':
                meta['initial_parameter_hash'] = agent.initial_parameter_hash
                meta['final_parameter_hash'] = agent.parameter_hash()
                meta['parameters_changed'] = agent.initial_parameter_hash != agent.parameter_hash()
                if episodes * EnvConfig().horizon >= agent.config.learning_starts and not meta['parameters_changed']:
                    raise RuntimeError('training did not change model parameters')
            agent.save(checkpoint, meta)
            meta['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            write_json(output / 'results' / f'manifest_{run_name}.json', meta)
    write_json(output / 'results' / status_name, {'status': 'COMPLETE', 'executed_at_utc': datetime.now(timezone.utc).isoformat(), 'seeds': list(seeds), 'episodes': episodes, 'algorithms': list(algorithms), 'observation_mode': observation_mode})


def checkpoint_policies(output, seed, include_no_memory=False):
    from .dqn import DQNAgent
    output = Path(output)
    paths = {'ddqn': output / f'models/ddqn_history_seed{seed}.pt',
             'untrained_ddqn': output / f'models/ddqn_history_seed{seed}_initial.pt',
             'tabular_q': output / f'models/tabular_q_seed{seed}.json'}
    if include_no_memory:
        paths['ddqn_no_memory'] = output / f'models/ddqn_no_memory_seed{seed}.pt'
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f'missing {path}; revised model execution is pending, or train this checkpoint first')
    policies = {name: TabularQAgent.load(path) if name == 'tabular_q' else DQNAgent.load(path) for name, path in paths.items()}
    for name, policy in policies.items():
        if policy.checkpoint_metadata.get('protocol', {}).get('environment') != asdict(EnvConfig()):
            raise ValueError(f'{name} checkpoint environment differs from the current protocol')
        if policy.checkpoint_metadata.get('source_sha256') != source_digest():
            raise ValueError(f'{name} checkpoint source hash differs; preserve and use the training source')
    policies.update({'no_op': NoOpPolicy(), 'alert_rule_diagnostic': AlertResponderPolicy()})
    return policies, {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def evaluate(output=ROOT, seeds=TRAINING_SEEDS, episodes=EVALUATION_EPISODES, include_no_memory=False, sensitivity=False):
    output = Path(output)
    prefix = 'sensitivity' if sensitivity else 'evaluation'
    intended = [output / 'results' / f'{prefix}_{suffix}' for suffix in ('episodes.csv', 'summary.csv', 'per_seed.csv', 'paired_differences.csv', 'manifest.json')]
    if any(path.exists() for path in intended):
        raise FileExistsError('Preserving previous evaluation evidence: choose a fresh output directory with the intended checkpoints')
    rows, hashes = [], {}
    conditions = SENSITIVITIES if sensitivity else (EnvConfig().alert_sensitivity,)
    for seed in seeds:
        policies, seed_hashes = checkpoint_policies(output, seed, include_no_memory)
        hashes[str(seed)] = seed_hashes
        for sensor in conditions:
            config = replace(EnvConfig(), alert_sensitivity=sensor)
            for episode in range(episodes):
                scenario = sensitivity_seed(episode) if sensitivity else evaluation_seed(episode)
                for name, fixed_policy in {**policies, 'random_legal': None}.items():
                    # Policy randomness is isolated from the environment's CRN fields.
                    policy = RandomPolicy(10000000 + seed * 1000 + episode) if name == 'random_legal' else fixed_policy
                    env = SentinelEnv(seed=scenario, config=config)
                    observation, info = env.reset(scenario)
                    for _ in range(config.horizon):
                        action = policy.act(observation, info['action_mask'], explore=False)
                        observation, _, terminated, truncated, info = env.step(action)
                        if terminated or truncated:
                            break
                    rows.append({'policy': name, 'training_seed': seed, 'scenario_index': episode,
                                 'environment_seed': scenario, 'alert_sensitivity': sensor, **env.metrics(),
                                 **{f'action_{a}': int(count) for a, count in enumerate(env.action_counts)}})
    prefix = 'sensitivity' if sensitivity else 'evaluation'
    write_csv(output / 'results' / f'{prefix}_episodes.csv', rows)
    summaries, differences = summarize(rows)
    seed_rows = []
    for sensor in conditions:
        for policy in sorted({row['policy'] for row in rows}):
            for seed in seeds:
                selected = [row for row in rows if row['alert_sensitivity'] == sensor and row['policy'] == policy and row['training_seed'] == seed]
                seed_rows.append({'policy': policy, 'alert_sensitivity': sensor, 'pairing_seed': seed,
                                  'scenarios': len(selected), **{metric + '_mean': float(np.mean([row[metric] for row in selected])) for metric in METRICS}})
    write_csv(output / 'results' / f'{prefix}_per_seed.csv', seed_rows)
    write_csv(output / 'results' / f'{prefix}_summary.csv', summaries)
    write_csv(output / 'results' / f'{prefix}_paired_differences.csv', differences)
    write_json(output / 'results' / f'{prefix}_manifest.json', {**metadata(), 'execution_status': 'COMPLETE',
            'actual_seeds': list(seeds), 'actual_episodes': episodes, 'checkpoint_hashes': hashes,
            'ci_method': 'crossed bootstrap: training-seed rows and common-scenario columns; 2000 replicates'})
    print(f'Wrote {len(rows)} real episode rows; no test-based checkpoint selection.', flush=True)


def summarize(rows):
    sensors = sorted({row['alert_sensitivity'] for row in rows})
    policies = sorted({row['policy'] for row in rows})
    seeds = sorted({row['training_seed'] for row in rows})
    scenarios = sorted({row['scenario_index'] for row in rows})
    index = {(row['alert_sensitivity'], row['policy'], row['training_seed'], row['scenario_index']): row for row in rows}
    summaries, differences = [], []
    for sensor in sensors:
        for metric in METRICS:
            values = {policy: np.array([[index[sensor, policy, seed, scenario][metric] for scenario in scenarios] for seed in seeds]) for policy in policies}
            for policy, value in values.items():
                mean, lo, hi = crossed_bootstrap(value)
                summaries.append({'policy': policy, 'alert_sensitivity': sensor, 'metric': metric, 'mean': mean,
                                  'ci95_low': lo, 'ci95_high': hi, 'seed_pairing_blocks': len(seeds),
                                  'independent_training_runs': len(seeds) if policy in ('ddqn', 'ddqn_no_memory', 'tabular_q') else 0,
                                  'seed_mean_sd': float(np.std(value.mean(axis=1), ddof=1)) if len(seeds) > 1 else 0.,
                                  'scenarios_per_seed': len(scenarios), 'episodes': value.size})
            if 'ddqn' in values:
                for baseline in policies:
                    if baseline == 'ddqn':
                        continue
                    mean, lo, hi = crossed_bootstrap(values['ddqn'] - values[baseline])
                    differences.append({'comparison': f'ddqn-minus-{baseline}', 'alert_sensitivity': sensor,
                                        'metric': metric, 'paired_mean_difference': mean, 'ci95_low': lo,
                                        'ci95_high': hi, 'seed_pairing_blocks': len(seeds), 'paired_scenarios': len(scenarios)})
    return summaries, differences
