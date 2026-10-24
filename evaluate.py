"""Standalone evaluation and plot generation script."""
import argparse
from pathlib import Path
import pandas as pd
from src.experiment import evaluate, evaluate_sensitivity, evaluate_ablation, make_plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Held-out evaluation episodes per baseline (default: 30)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    train_path = root / "results" / "training_episodes.csv"
    if not train_path.exists():
        raise SystemExit("Training log missing. Run `python train.py` or `python run_all.py` first.")

    training = pd.read_csv(train_path)
    evaluation = evaluate(episodes=args.episodes)
    sensitivity = evaluate_sensitivity(episodes=20)
    ablation = evaluate_ablation(episodes=args.episodes)
    make_plots(training, evaluation, sensitivity, ablation)
    print("\n=== Held-Out Benchmark Summary ===")
    print(pd.read_csv(root / "results" / "summary.csv").to_string(index=False))
    print("\n=== Attack-Rate Sensitivity Summary ===")
    print(pd.read_csv(root / "results" / "sensitivity_summary.csv").to_string(index=False))
    print("\n=== Ablation & RL Comparator Summary ===")
    print(pd.read_csv(root / "results" / "ablation_summary.csv").to_string(index=False))
