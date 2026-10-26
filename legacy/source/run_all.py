"""One-command end-to-end experiment runner."""
import argparse
from src.experiment import run_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=280, help="Training episodes per DQN seed")
    parser.add_argument("--eval-episodes", type=int, default=30, help="Held-out evaluation episodes per baseline")
    args = parser.parse_args()
    if args.episodes < 5 or args.eval_episodes < 5:
        parser.error("Use at least 5 training and 5 evaluation episodes for a meaningful smoke run")
    run_all(episodes=args.episodes, eval_episodes=args.eval_episodes)
