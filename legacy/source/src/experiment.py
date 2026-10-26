"""Training, held-out evaluation, sensitivity analysis, ablation study, and figure generation."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
except ImportError:  # pragma: no cover
    from ._torch_compat import torch  # type: ignore

from .environment import (
    SentinelEnv,
    EnvConfig,
    N_ACTIONS,
    OBS_DIM,
    CRITICALITY,
    WAIT,
    INSPECT_BASE,
    PATCH_BASE,
    ISOLATE_BASE,
    DECOY_BASE,
)
from .dqn import DQNAgent, DQNConfig
from .policies import (
    RandomLegalPolicy,
    PatchPriorityPolicy,
    AlertResponderPolicy,
    ThresholdDefensePolicy,
    TabularQLearningAgent,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def train_one(seed: int, episodes: int = 280, config: DQNConfig | None = None) -> tuple[Path, list[dict]]:
    env_cfg = EnvConfig()
    agent_cfg = config or DQNConfig(
        train_every=8,
        epsilon_decay_steps=max(6500, int(episodes * env_cfg.horizon * 0.65)),
    )
    env = SentinelEnv(seed=seed, config=env_cfg)
    agent = DQNAgent(OBS_DIM, N_ACTIONS, seed=seed, config=agent_cfg, device="cpu")

    # 1. Seed demonstration buffer with expert trajectories for DQfD margin regularization
    expert = AlertResponderPolicy()
    for ep in range(25):
        expert_env = SentinelEnv(seed=seed * 50000 + ep, config=env_cfg)
        obs, info = expert_env.reset(seed=seed * 50000 + ep)
        done = False
        while not done:
            mask = info["action_mask"]
            a = expert.act(obs, mask)
            next_obs, r, term, trunc, next_info = expert_env.step(a)
            done = term or trunc
            agent.add_expert(obs, a, r, next_obs, done, next_info["action_mask"])
            obs, info = next_obs, next_info

    # 2. Main RL training loop with n-step returns and Double-DQN updates
    rows: list[dict] = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed * 10000 + ep)
        done = False
        ep_return = 0.0
        losses: list[float] = []
        while not done:
            mask = info["action_mask"]
            action = agent.act(obs, mask, explore=True)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            agent.steps += 1

            # Containment-aligned training shaping (true unshaped reward is logged for episode_return)
            train_r = reward
            if ISOLATE_BASE <= action < DECOY_BASE and next_info["isolated_true"]:
                host = action - ISOLATE_BASE
                if CRITICALITY[host] >= 0.8:
                    train_r += 1.2
            if ISOLATE_BASE <= action < DECOY_BASE and next_info["isolated_false"]:
                train_r -= 2.5
            if done and next_info["active_compromised_at_end"] > 0:
                train_r -= 1.0 * next_info["critical_compromises_total"]

            agent.remember(obs, action, train_r, next_obs, done, next_info["action_mask"])
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            obs, info = next_obs, next_info
            ep_return += reward

        summary = env.summary()
        rows.append(
            {
                "seed": seed,
                "episode": ep + 1,
                "episode_return": ep_return,
                "loss_mean": float(np.mean(losses)) if losses else np.nan,
                "epsilon": agent.epsilon(),
                "total_compromises": summary["total_compromises"],
                "critical_compromises": summary["critical_compromises"],
                "active_compromised_at_end": summary["active_compromised_at_end"],
                "mean_detection_delay": summary["mean_detection_delay"],
                "false_isolations": summary["false_isolations"],
                "budget_used": summary["budget_used"],
            }
        )
        if (ep + 1) % 50 == 0:
            print(f"seed={seed} episode={ep+1}/{episodes} return={ep_return:.2f} eps={agent.epsilon():.3f}")

    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / f"training_seed{seed}.csv", index=False)
    ckpt = MODELS / f"sentinel_dqn_seed{seed}.pt"
    agent.save(str(ckpt), metadata={"seed": seed, "episodes": episodes, "env": env_cfg.__dict__})
    return ckpt, rows


def _run_episode(
    policy_name: str,
    policy,
    seed: int,
    model_seed: int | str | None = None,
    config: EnvConfig | None = None,
) -> dict:
    env = SentinelEnv(seed=seed, config=config or EnvConfig())
    obs, info = env.reset(seed=seed)
    done = False
    total_return = 0.0
    action_counts = {"wait": 0, "inspect": 0, "patch": 0, "isolate": 0, "decoy": 0}
    while not done:
        mask = info["action_mask"]
        if isinstance(policy, DQNAgent):
            action = policy.act(obs, mask, explore=False)
        else:
            action = policy.act(obs, mask)
        if action == 0:
            action_counts["wait"] += 1
        elif action < 1 + 6:
            action_counts["inspect"] += 1
        elif action < 1 + 12:
            action_counts["patch"] += 1
        elif action < 1 + 18:
            action_counts["isolate"] += 1
        else:
            action_counts["decoy"] += 1
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_return += reward
    row = {
        "policy": policy_name,
        "model_seed": model_seed if model_seed is not None else "baseline",
        "eval_seed": seed,
        **env.summary(),
    }
    row.update({f"actions_{k}": v for k, v in action_counts.items()})
    return row


def evaluate(episodes: int = 30, model_seeds: tuple[int, ...] = (17, 29, 43)) -> pd.DataFrame:
    rows: list[dict] = []
    eval_seeds = [300000 + i * 97 for i in range(episodes)]
    for seed in model_seeds:
        ckpt = MODELS / f"sentinel_dqn_seed{seed}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint {ckpt}; run training first")
        agent = DQNAgent.load(str(ckpt), device="cpu")
        for es in eval_seeds:
            rows.append(_run_episode("DQN", agent, es, model_seed=seed))

    baseline_factories = [
        ("Random legal", lambda s: RandomLegalPolicy(s)),
        ("Patch priority", lambda s: PatchPriorityPolicy()),
        ("Alert responder", lambda s: AlertResponderPolicy()),
        ("Threshold defense", lambda s: ThresholdDefensePolicy()),
    ]
    for name, factory in baseline_factories:
        for es in eval_seeds:
            rows.append(_run_episode(name, factory(es), es))

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "evaluation_episodes.csv", index=False)

    metric_cols = [
        "episode_return",
        "total_compromises",
        "critical_compromises",
        "active_compromised_at_end",
        "mean_detection_delay",
        "false_isolations",
        "successful_isolations",
        "budget_used",
        "remaining_budget",
        "detected_hosts",
    ]
    dqn_rows = df[df["policy"] == "DQN"].groupby("eval_seed")[metric_cols].mean().reset_index()
    dqn_rows["policy"] = "DQN"
    baseline_rows = df[df["policy"] != "DQN"][["policy", "eval_seed", *metric_cols]]
    scenario_level = pd.concat(
        [baseline_rows, dqn_rows[["policy", "eval_seed", *metric_cols]]], ignore_index=True
    )
    summary = scenario_level.groupby("policy")[metric_cols].agg(["mean", "std"]).round(3)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(RESULTS / "summary.csv", index=False)
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "episodes_per_baseline": episodes,
                "dqn_training_seeds": list(model_seeds),
                "dqn_evaluation_runs": episodes * len(model_seeds),
                "metrics": summary.to_dict(orient="records"),
            },
            f,
            indent=2,
        )
    return df


def evaluate_sensitivity(episodes: int = 20, model_seeds: tuple[int, ...] = (17, 29, 43)) -> pd.DataFrame:
    """Evaluate generalisation under attack-rate shifts without retraining."""
    rows = []
    attack_rates = (0.18, 0.23, 0.28)
    for rate_idx, rate in enumerate(attack_rates):
        cfg = EnvConfig(
            response_budget=6,
            attack_base_probability=rate,
            alert_sensitivity=0.55,
            false_positive_rate=0.08,
        )
        seeds = [400000 + rate_idx * 10000 + i * 53 for i in range(episodes)]
        for model_seed in model_seeds:
            agent = DQNAgent.load(str(MODELS / f"sentinel_dqn_seed{model_seed}.pt"), device="cpu")
            for es in seeds:
                row = _run_episode("DQN", agent, es, model_seed=model_seed, config=cfg)
                row["attack_probability"] = rate
                rows.append(row)
        for name, factory in (
            ("Alert responder", lambda s: AlertResponderPolicy()),
            ("Threshold defense", lambda s: ThresholdDefensePolicy()),
        ):
            for es in seeds:
                row = _run_episode(name, factory(es), es, config=cfg)
                row["attack_probability"] = rate
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "sensitivity_episodes.csv", index=False)
    sens_metrics = [
        "episode_return",
        "critical_compromises",
        "active_compromised_at_end",
        "mean_detection_delay",
    ]
    dqn_rows = (
        df[df["policy"] == "DQN"]
        .groupby(["attack_probability", "eval_seed"])[sens_metrics]
        .mean()
        .reset_index()
    )
    dqn_rows["policy"] = "DQN"
    baseline_rows = df[df["policy"] != "DQN"][["policy", "attack_probability", "eval_seed", *sens_metrics]]
    scenario_level = pd.concat(
        [baseline_rows, dqn_rows[["policy", "attack_probability", "eval_seed", *sens_metrics]]],
        ignore_index=True,
    )
    summary = scenario_level.groupby(["policy", "attack_probability"])[sens_metrics].agg(["mean", "std"]).round(3)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(RESULTS / "sensitivity_summary.csv", index=False)
    return df


def evaluate_ablation(episodes: int = 30, model_seeds: tuple[int, ...] = (17, 29, 43)) -> pd.DataFrame:
    """Compare Full Graph-Dueling DQN against architectural ablations and Tabular Q-Learning."""
    eval_seeds = [300000 + i * 97 for i in range(episodes)]
    rows = []

    # 1. Full Model (Graph-Dueling DDQN + Stealth-Transition Prior)
    for seed in model_seeds:
        agent = DQNAgent.load(str(MODELS / f"sentinel_dqn_seed{seed}.pt"), device="cpu")
        for es in eval_seeds:
            rows.append(_run_episode("Graph-Dueling DQN (Full)", agent, es, model_seed=seed))

    # 2. Ablation: Dueling DDQN without Stealth-Transition Detector
    for seed in model_seeds:
        agent = DQNAgent.load(str(MODELS / f"sentinel_dqn_seed{seed}.pt"), device="cpu")
        agent.config.use_stealth_detector = False
        agent.online.use_stealth_detector = False
        for es in eval_seeds:
            rows.append(_run_episode("Dueling DQN (No Stealth Prior)", agent, es, model_seed=seed))

    # 3. Classic RL Comparator: Tabular Q-Learning on Discretized Belief States
    for seed in model_seeds:
        tab_agent = TabularQLearningAgent(seed=seed)
        train_env = SentinelEnv(seed=seed)
        for ep in range(280):
            obs, info = train_env.reset(seed=seed * 10000 + ep)
            done = False
            eps = max(0.05, 0.85 * (1.0 - ep / 220.0))
            while not done:
                mask = info["action_mask"]
                a = tab_agent.act(obs, mask, epsilon=eps)
                n_obs, r, term, trunc, n_info = train_env.step(a)
                done = term or trunc
                tab_agent.update(obs, a, r, n_obs, done, n_info["action_mask"])
                obs, info = n_obs, n_info
        for es in eval_seeds:
            rows.append(_run_episode("Tabular Q-Learning", tab_agent, es, model_seed=seed))

    df = pd.DataFrame(rows)
    metric_cols = [
        "episode_return",
        "critical_compromises",
        "active_compromised_at_end",
        "mean_detection_delay",
        "false_isolations",
        "successful_isolations",
    ]
    scen = df.groupby(["policy", "eval_seed"])[metric_cols].mean().reset_index()
    summary = scen.groupby("policy")[metric_cols].agg(["mean", "std"]).round(3)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS / "ablation_summary.csv", index=False)
    return df


def make_plots(
    training_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame | None = None,
    ablation_df: pd.DataFrame | None = None,
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    palette = {
        "DQN": "#1b6ca8",
        "Alert responder": "#2a9d8f",
        "Threshold defense": "#e9c46a",
        "Patch priority": "#f4a261",
        "Random legal": "#e76f51",
    }

    # 1. Training Curve
    plt.figure(figsize=(7.2, 4.1))
    seed_colors = {17: "#1b6ca8", 29: "#2a9d8f", 43: "#e76f51"}
    for seed, group in training_df.groupby("seed"):
        vals = group["episode_return"].rolling(20, min_periods=1).mean()
        plt.plot(
            group["episode"],
            vals,
            label=f"DQN Seed {seed}",
            linewidth=1.8,
            color=seed_colors.get(int(seed), None),
        )
    plt.axhline(-2.573, color="#333333", linestyle="--", linewidth=1.2, label="Alert Responder (-2.57)")
    plt.xlabel("Training Episode")
    plt.ylabel("20-Episode Rolling Return")
    plt.title("SentinelRL Graph-Dueling DQN Training Convergence")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS / "training_curve.png", dpi=200)
    plt.close()

    # 2. Main Evaluation Bar Plots
    metrics = [
        ("critical_compromises", "Critical-host compromises per episode", "critical_compromises.png"),
        ("active_compromised_at_end", "Uncontained compromises at episode end", "uncontained_at_end.png"),
        ("mean_detection_delay", "Mean detection delay (simulated steps)", "detection_delay.png"),
        ("false_isolations", "Unnecessary isolations per episode", "false_isolations.png"),
    ]
    preferred = ["DQN", "Alert responder", "Threshold defense", "Patch priority", "Random legal"]
    plot_metrics = [m[0] for m in metrics]
    dqn_plot = eval_df[eval_df["policy"] == "DQN"].groupby("eval_seed")[plot_metrics].mean().reset_index()
    dqn_plot["policy"] = "DQN"
    base_plot = eval_df[eval_df["policy"] != "DQN"][["policy", "eval_seed", *plot_metrics]]
    eval_plot_df = pd.concat(
        [base_plot, dqn_plot[["policy", "eval_seed", *plot_metrics]]], ignore_index=True
    )

    for col, ylabel, fname in metrics:
        stats = eval_plot_df.groupby("policy")[col].agg(["mean", "std"]).reindex(preferred).dropna()
        plt.figure(figsize=(7.2, 4.1))
        bar_colors = [palette.get(idx, "#4a6fe3") for idx in stats.index]
        means = stats["mean"].to_numpy(dtype=float)
        stds = stats["std"].fillna(0).to_numpy(dtype=float)
        yerr_low = np.minimum(np.maximum(0.0, means), stds)
        bars = plt.bar(
            stats.index,
            means,
            yerr=[yerr_low, stds],
            capsize=4,
            color=bar_colors,
            edgecolor="#222222",
            linewidth=0.7,
        )
        y_top = float(np.max(means + stds)) if len(means) else 1.0
        plt.ylim(0.0, max(0.25, y_top * 1.15))
        for bar, mean_val in zip(bars, means):
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                max(0.0, float(mean_val)) + max(0.015, y_top * 0.03),
                f"{mean_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.88),
            )
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.xticks(rotation=15, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(RESULTS / fname, dpi=200)
        plt.close()

    # 3. Sensitivity Plot
    if sensitivity_df is not None:
        sens_dqn = (
            sensitivity_df[sensitivity_df["policy"] == "DQN"]
            .groupby(["attack_probability", "eval_seed"])[["critical_compromises"]]
            .mean()
            .reset_index()
        )
        sens_dqn["policy"] = "DQN"
        sens_base = sensitivity_df[sensitivity_df["policy"] != "DQN"][
            ["policy", "attack_probability", "eval_seed", "critical_compromises"]
        ]
        sens_plot_df = pd.concat(
            [sens_base, sens_dqn[["policy", "attack_probability", "eval_seed", "critical_compromises"]]],
            ignore_index=True,
        )
        sub = (
            sens_plot_df.groupby(["policy", "attack_probability"])["critical_compromises"]
            .agg(["mean", "std"])
            .reset_index()
        )
        plt.figure(figsize=(7.0, 4.0))
        markers = {"DQN": "o", "Alert responder": "s", "Threshold defense": "^"}
        for policy_name in ("DQN", "Alert responder", "Threshold defense"):
            p = sub[sub["policy"] == policy_name].sort_values("attack_probability")
            if not p.empty:
                p_means = p["mean"].to_numpy(dtype=float)
                p_stds = p["std"].fillna(0).to_numpy(dtype=float)
                plt.errorbar(
                    p["attack_probability"],
                    p_means,
                    yerr=[np.minimum(np.maximum(0.0, p_means), p_stds), p_stds],
                    marker=markers.get(policy_name, "o"),
                    linewidth=2.0,
                    capsize=4,
                    label=policy_name,
                    color=palette.get(policy_name, None),
                )
        plt.xlabel("Attack base probability")
        plt.ylabel("Critical-host compromises per episode")
        plt.title("Out-of-Distribution Robustness Across Attack Intensities")
        plt.grid(alpha=0.25)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(RESULTS / "sensitivity_critical_compromises.png", dpi=200)
        plt.close()

    # 4. Ablation Plot
    if ablation_df is not None:
        scen = (
            ablation_df.groupby(["policy", "eval_seed"])[
                ["episode_return", "critical_compromises", "active_compromised_at_end"]
            ]
            .mean()
            .reset_index()
        )
        order = ["Graph-Dueling DQN (Full)", "Dueling DQN (No Stealth Prior)", "Tabular Q-Learning"]
        stats = scen.groupby("policy")[["critical_compromises", "active_compromised_at_end"]].mean().reindex(order)
        x = np.arange(len(order))
        width = 0.35
        plt.figure(figsize=(7.2, 4.0))
        plt.bar(x - width / 2, stats["critical_compromises"], width, label="Critical Compromises", color="#1b6ca8")
        plt.bar(x + width / 2, stats["active_compromised_at_end"], width, label="Uncontained at End", color="#e76f51")
        plt.xticks(x, order, rotation=12, ha="right")
        plt.ylabel("Mean Hosts per Episode (Lower is Better)")
        plt.title("RL Architecture & Algorithm Ablation Study")
        plt.grid(axis="y", alpha=0.25)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(RESULTS / "ablation_comparison.png", dpi=200)
        plt.close()


def run_all(episodes: int = 280, eval_episodes: int = 30, seeds: tuple[int, ...] = (17, 29, 43)):
    torch.set_num_threads(1)
    all_training = []
    for seed in seeds:
        _, rows = train_one(seed=seed, episodes=episodes)
        all_training.extend(rows)
    training_df = pd.DataFrame(all_training)
    training_df.to_csv(RESULTS / "training_episodes.csv", index=False)
    eval_df = evaluate(episodes=eval_episodes, model_seeds=seeds)
    sensitivity_df = evaluate_sensitivity(episodes=20, model_seeds=seeds)
    ablation_df = evaluate_ablation(episodes=eval_episodes, model_seeds=seeds)
    make_plots(training_df, eval_df, sensitivity_df, ablation_df)
    print("\nEvaluation summary:")
    print(pd.read_csv(RESULTS / "summary.csv").to_string(index=False))
    print("\nAttack-rate sensitivity summary:")
    print(pd.read_csv(RESULTS / "sensitivity_summary.csv").to_string(index=False))
    print("\nAblation summary:")
    print(pd.read_csv(RESULTS / "ablation_summary.csv").to_string(index=False))
    return training_df, eval_df
