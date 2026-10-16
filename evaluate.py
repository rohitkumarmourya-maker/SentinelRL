from src.experiment import evaluate, make_plots
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    train_path = root / "results" / "training_episodes.csv"
    if not train_path.exists():
        raise SystemExit("Training log missing. Run python train.py first.")
    training = pd.read_csv(train_path)
    episodes = int(input("Held-out evaluation episodes per baseline [60]: ") or "60")
    evaluation = evaluate(episodes=episodes)
    make_plots(training, evaluation)
    print(pd.read_csv(root / "results" / "summary.csv").to_string(index=False))
