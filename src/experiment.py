"""Training, held-out evaluation, tabulation and figure generation."""
from __future__ import annotations
import csv, json, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from .environment import SentinelEnv, EnvConfig, N_ACTIONS, OBS_DIM
from .dqn import DQNAgent, DQNConfig
from .policies import RandomLegalPolicy, PatchPriorityPolicy, AlertResponderPolicy, ThresholdDefensePolicy

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def train_one(seed: int, episodes: int, config: DQNConfig | None = None) -> tuple[Path, list[dict]]:
    env_cfg = EnvConfig()
    agent_cfg = config or DQNConfig(train_every=4, epsilon_decay_steps=max(6000, int(episodes * env_cfg.horizon * 0.70)))
    env = SentinelEnv(seed=seed, config=env_cfg)
    agent = DQNAgent(OBS_DIM, N_ACTIONS, seed=seed, config=agent_cfg, device="cpu")
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
            agent.remember(obs, action, reward, next_obs, done, next_info["action_mask"])
            loss = agent.learn()
            if loss is not None: losses.append(loss)
            obs, info = next_obs, next_info
            ep_return += reward
        summary = env.summary()
        rows.append({"seed": seed, "episode": ep + 1, "episode_return": ep_return,
                     "loss_mean": float(np.mean(losses)) if losses else np.nan,
                     "epsilon": agent.epsilon(), "total_compromises": summary["total_compromises"],
                     "critical_compromises": summary["critical_compromises"],
                     "active_compromised_at_end": summary["active_compromised_at_end"],
                     "mean_detection_delay": summary["mean_detection_delay"],
                     "false_isolations": summary["false_isolations"],
                     "budget_used": summary["budget_used"]})
        if (ep + 1) % 50 == 0:
            print(f"seed={seed} episode={ep+1}/{episodes} return={ep_return:.2f} eps={agent.epsilon():.3f}")
    MODELS.mkdir(parents=True, exist_ok=True)
    ckpt = MODELS / f"sentinel_dqn_seed{seed}.pt"
    agent.save(str(ckpt), metadata={"seed": seed, "episodes": episodes, "env": env_cfg.__dict__})
    return ckpt, rows


def _run_episode(policy_name: str, policy, seed: int, model_seed: int | None = None, config: EnvConfig | None = None) -> dict:
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
        if action == 0: action_counts["wait"] += 1
        elif action < 1 + 6: action_counts["inspect"] += 1
        elif action < 1 + 12: action_counts["patch"] += 1
        elif action < 1 + 18: action_counts["isolate"] += 1
        else: action_counts["decoy"] += 1
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_return += reward
    row = {"policy": policy_name, "model_seed": model_seed if model_seed is not None else "baseline",
           "eval_seed": seed, **env.summary()}
    row.update({f"actions_{k}": v for k, v in action_counts.items()})
    return row


def evaluate(episodes: int = 60, model_seeds: tuple[int, ...] = (17, 29, 43)) -> pd.DataFrame:
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
    metric_cols = ["episode_return", "total_compromises", "critical_compromises", "active_compromised_at_end",
                   "mean_detection_delay", "false_isolations", "successful_isolations", "budget_used",
                   "remaining_budget", "detected_hosts"]
    # Average the three DQN checkpoints within each held-out scenario first, so the
    # DQN summary and each baseline summary are both based on 30 scenario-level rows.
    dqn_rows = df[df["policy"] == "DQN"].groupby("eval_seed")[metric_cols].mean().reset_index()
    dqn_rows["policy"] = "DQN"
    baseline_rows = df[df["policy"] != "DQN"][["policy", "eval_seed", *metric_cols]]
    scenario_level = pd.concat([baseline_rows, dqn_rows[["policy", "eval_seed", *metric_cols]]], ignore_index=True)
    summary = scenario_level.groupby("policy")[metric_cols].agg(["mean", "std"]).round(3)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(RESULTS / "summary.csv", index=False)
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"episodes_per_baseline": episodes, "dqn_training_seeds": list(model_seeds),
                   "dqn_evaluation_runs": episodes * len(model_seeds),
                   "metrics": summary.to_dict(orient="records")}, f, indent=2)
    return df


def make_plots(training_df: pd.DataFrame, eval_df: pd.DataFrame, sensitivity_df: pd.DataFrame | None = None) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.0))
    for seed, group in training_df.groupby("seed"):
        vals = group["episode_return"].rolling(20, min_periods=1).mean()
        plt.plot(group["episode"], vals, label=f"Seed {seed}", linewidth=1.5)
    plt.xlabel("Training episode"); plt.ylabel("20-episode rolling return")
    plt.title("SentinelRL DQN training"); plt.grid(alpha=0.25); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(RESULTS / "training_curve.png", dpi=180); plt.close()

    metrics = [("critical_compromises", "Critical-host compromises per episode", "critical_compromises.png"),
               ("active_compromised_at_end", "Uncontained compromises at episode end", "uncontained_at_end.png"),
               ("mean_detection_delay", "Mean detection delay (simulated steps)", "detection_delay.png"),
               ("false_isolations", "Unnecessary isolations per episode", "false_isolations.png")]
    preferred = ["DQN", "Alert responder", "Threshold defense", "Patch priority", "Random legal"]
    plot_metrics = [m[0] for m in metrics]
    dqn_plot = eval_df[eval_df["policy"] == "DQN"].groupby("eval_seed")[plot_metrics].mean().reset_index()
    dqn_plot["policy"] = "DQN"
    base_plot = eval_df[eval_df["policy"] != "DQN"][["policy", "eval_seed", *plot_metrics]]
    eval_plot_df = pd.concat([base_plot, dqn_plot[["policy", "eval_seed", *plot_metrics]],], ignore_index=True)
    for col, ylabel, fname in metrics:
        stats = eval_plot_df.groupby("policy")[col].agg(["mean", "std"]).reindex(preferred).dropna()
        plt.figure(figsize=(7.2, 4.1))
        plt.bar(stats.index, stats["mean"], yerr=stats["std"].fillna(0), capsize=3)
        plt.ylabel(ylabel); plt.title(ylabel); plt.xticks(rotation=18, ha="right"); plt.grid(axis="y", alpha=0.25)
        plt.tight_layout(); plt.savefig(RESULTS / fname, dpi=180); plt.close()
    if sensitivity_df is not None:
        sens_dqn = sensitivity_df[sensitivity_df["policy"] == "DQN"].groupby(["attack_probability", "eval_seed"])[["critical_compromises"]].mean().reset_index()
        sens_dqn["policy"] = "DQN"
        sens_base = sensitivity_df[sensitivity_df["policy"] != "DQN"][["policy", "attack_probability", "eval_seed", "critical_compromises"]]
        sens_plot_df = pd.concat([sens_base, sens_dqn[["policy", "attack_probability", "eval_seed", "critical_compromises"]]], ignore_index=True)
        sub = sens_plot_df.groupby(["policy", "attack_probability"])["critical_compromises"].agg(["mean", "std"]).reset_index()
        plt.figure(figsize=(7.0, 4.0))
        for policy_name in ("DQN", "Alert responder", "Threshold defense"):
            p = sub[sub["policy"] == policy_name].sort_values("attack_probability")
            if not p.empty:
                plt.errorbar(p["attack_probability"], p["mean"], yerr=p["std"].fillna(0), marker="o", capsize=3, label=policy_name)
        plt.xlabel("Attack success probability multiplier")
        plt.ylabel("Critical-host compromises per episode")
        plt.title("Sensitivity to attack intensity"); plt.grid(alpha=0.25); plt.legend(frameon=False)
        plt.tight_layout(); plt.savefig(RESULTS / "sensitivity_critical_compromises.png", dpi=180); plt.close()


def evaluate_sensitivity(episodes: int = 20, model_seeds: tuple[int, ...] = (17, 29, 43)) -> pd.DataFrame:
    """Evaluate generalisation under attack-rate shifts; no additional training."""
    rows = []
    attack_rates = (0.18, 0.23, 0.28)
    for rate_idx, rate in enumerate(attack_rates):
        cfg = EnvConfig(response_budget=6, attack_base_probability=rate, alert_sensitivity=0.55, false_positive_rate=0.08)
        seeds = [400000 + rate_idx * 10000 + i * 53 for i in range(episodes)]
        for model_seed in model_seeds:
            agent = DQNAgent.load(str(MODELS / f"sentinel_dqn_seed{model_seed}.pt"), device="cpu")
            for es in seeds:
                row = _run_episode("DQN", agent, es, model_seed=model_seed, config=cfg)
                row["attack_probability"] = rate
                rows.append(row)
        for name, factory in (("Alert responder", lambda s: AlertResponderPolicy()), ("Threshold defense", lambda s: ThresholdDefensePolicy())):
            for es in seeds:
                row = _run_episode(name, factory(es), es, config=cfg)
                row["attack_probability"] = rate
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "sensitivity_episodes.csv", index=False)
    sens_metrics = ["episode_return", "critical_compromises", "active_compromised_at_end", "mean_detection_delay"]
    dqn_rows = df[df["policy"] == "DQN"].groupby(["attack_probability", "eval_seed"])[sens_metrics].mean().reset_index()
    dqn_rows["policy"] = "DQN"
    baseline_rows = df[df["policy"] != "DQN"][["policy", "attack_probability", "eval_seed", *sens_metrics]]
    scenario_level = pd.concat([baseline_rows, dqn_rows[["policy", "attack_probability", "eval_seed", *sens_metrics]]], ignore_index=True)
    summary = scenario_level.groupby(["policy", "attack_probability"])[sens_metrics].agg(["mean", "std"]).round(3)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(RESULTS / "sensitivity_summary.csv", index=False)
    return df


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
    make_plots(training_df, eval_df, sensitivity_df)
    print("\nEvaluation summary:")
    print(pd.read_csv(RESULTS / "summary.csv").to_string(index=False))
    print("\nAttack-rate sensitivity summary:")
    print(pd.read_csv(RESULTS / "sensitivity_summary.csv").to_string(index=False))
    return training_df, eval_df
