# Assignment deliverables and current status

| Requirement | Completed evidence |
|---|---|
| Reinforcement learning as central solution | Pure observation-only Double DQN; no heuristic teacher, fixed Q prior or policy shield |
| Stochastic sequential decision problem | Six-host POMDP, noisy sensors, six response tokens, 48-step finite horizon |
| Complete runnable source/setup/dependencies | Included; no external dataset/credentials/GPU/service required |
| Trained RL agent | Three final primary neural checkpoints, all seeds 17/29/43 |
| Independent seeds and real gradients | Three seeds per algorithm/condition; every DDQN 29,751 updates and changed parameter hashes |
| Baselines and ablations | Tabular Q, initial network, random legal, no-op, diagnostic alert rule; independently retrained no-memory ablation |
| Held-out comparisons and uncertainty |2100 primary rows; 6300 sensor rows; per-seed means/SD, crossed bootstrap intervals and within-pair differences |
| Graphs | Actual learning curve and policy comparison, report-specific figures and raw underlying data |
| Team and GitHub | At top of README and report |
| Exactly two pages/two columns | `report/main.pdf`, including references; editable TeX/figure sources |
| S/O dimensions, A, mathematical R, dynamics, algorithm | Rigorous report plus design docs |
| Genuine history/provenance | Original five commits preserved unchanged; current source/run explicitly disclosed with hashes |
| Tests and runnable pretrained demonstration | All 35 tests passed; saved checkpoint demo and exact raw episode replay passed |
| Honest limitations | No supported primary advantage over alert rule; low successful confirmation coverage; all seeds and controls retained |

The initial model-run hold was lifted by the user's explicit “run now” instruction. The earlier pending-results package remains separate; this is the actual trained delivery. No GitHub push was performed. Performance claims are bounded to this small simulator; a strong result is not manufactured by changing the task or hiding comparators that win.
