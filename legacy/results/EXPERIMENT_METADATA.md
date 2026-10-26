# Experiment metadata and execution record

- Project: SentinelRL (custom abstract defensive cyber-range POMDP)
- Run date: 2026-10-24
- Interpreter: Python 3.11+ / 3.13 compatible
- Compute: CPU only; no GPU required
- Test result after final source changes: `8 passed`
- Training: three independent seeds (17, 29, 43), 280 episodes each, 13,440 environment steps each
- Standard evaluation: 30 held-out scenario seeds per baseline (`300000 + 97*i`); three DQN checkpoints on the same 30 scenarios (90 DQN checkpoint/scenario episodes). Summary statistics average DQN checkpoint outcomes within each scenario before measuring across-scenario mean and standard deviation (`DQN`: return `-0.360 ± 2.672`, critical compromises `0.111 ± 0.364`, uncontained `0.044 ± 0.243`, detection delay `0.922 ± 3.677`, false isolations `0.000 ± 0.000`).
- Attack-rate sensitivity: 20 held-out scenarios per condition for base probabilities 0.18, 0.23, 0.28 (`400000 + 10000*rate_idx + 53*i`), comparing three DQN checkpoints, Alert Responder and Threshold Defense.
- Architectural & RL algorithm ablation: 30 held-out scenarios comparing `Graph-Dueling DQN (Full)`, `Dueling DQN (No Stealth Prior)`, and `Tabular Q-Learning` (`results/ablation_summary.csv`).
- PDF report: compiled in two-column format (`report/main.pdf`); strictly two main report pages plus references on page 3.
