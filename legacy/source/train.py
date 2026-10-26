from src.experiment import train_one
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    rows_all = []
    for seed in (17, 29, 43):
        path, rows = train_one(seed=seed, episodes=280)
        print(f"Saved checkpoint: {path}")
        rows_all.extend(rows)
    pd.DataFrame(rows_all).to_csv(Path(__file__).resolve().parent / "results" / "training_episodes.csv", index=False)
