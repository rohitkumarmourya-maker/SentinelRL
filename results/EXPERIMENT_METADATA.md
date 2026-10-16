# Experiment metadata and execution record

- Project: SentinelRL (custom abstract defensive cyber POMDP)
- Run date: 2026-10-09
- Interpreter: Python 3.13.5
- NumPy: 2.3.5
- pandas: 2.2.3
- Matplotlib: 3.10.8
- PyTorch: 2.10.0+cpu
- pytest: 9.0.2
- Compute: CPU only; no GPU was used
- Test result after final source changes: `6 passed`
- Training: three independent seeds (17, 29, 43), 280 episodes each, 13,440 environment steps each, 3,236 optimizer updates each
- Standard evaluation: 30 held-out scenario seeds per baseline; three DQN checkpoints on the same 30 scenarios (90 DQN checkpoint/scenario episodes). Summary statistics average DQN checkpoint outcomes within each scenario before measuring across-scenario mean and standard deviation.
- Attack-rate sensitivity: 20 held-out scenarios per condition for base probabilities 0.18, 0.23, 0.28, comparing three DQN checkpoints, Alert Responder and Threshold Defense.
- PDF report: compiled and rendered for visual inspection; three pages total, comprising two main report pages plus references.

The standard evaluation and sensitivity results were run after training. The report's mean/std values are taken from `summary.csv` and `sensitivity_summary.csv`; plotted results were regenerated from the corresponding episode CSVs using scenario-level aggregation for DQN. No significance tests, live-network testing or real-world validation were performed.
