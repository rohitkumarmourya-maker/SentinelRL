# Validation record — 2026-10-26

The initial delivery honored the user's request not to run models. The user subsequently said **“run now”**, authorizing the actual model checks, training, inference and evaluation recorded here. Earlier pending-results ZIP/PDF files were preserved; this trained package contains the completed evidence.

## Executed checks and experiment

| Command / check | Actual outcome |
|---|---|
| `python3 -m pytest -o addopts='' -m model -q` | 4 model tests passed, 31 non-model deselected; 1.18 s |
| `python3 -m pytest -q` after removing model exclusion | **35 passed**, 0.82 s; includes real gradients/checkpoint inference |
| Python compilation / static AST checks | Passed |
| Full `run_all.py --execute-model-run --output runs/full_20261026` | All 9 training runs,2100 primary and 6300 sensor-evaluation rows completed; plots generated |
| Saved parameter/change and checkpoint/source hashes | All 6 trained neural models changed; each 29,751 gradient updates; all 9 run manifests complete |
| `python3 scripts/verify_evidence.py --replay` | All raw count/finite-value/hash/legality checks passed; saved seed 17/scenario 300000 exactly reproduced all metrics and 25 action counts |
| `python3 demo.py --checkpoint models/ddqn_history_seed17.pt --execute-model-run --steps 8` | Trained-policy inference passed using bundled checkpoint |
| Original ZIP vs clone / preserved source hashes | All 47 original files preserved exactly; original complete Git bundle verified |

The simulator checks cover hidden-state noninterference, initial entry leakage, stale diagnoses, simultaneous spread, residual defense risk, isolation edges, first-only diagnostic rewards, explicit delay cost, false-isolation downtime, budget/invalid actions, finite horizon, censoring and action-independent common randomness. The model checks verify actual gradient changes/target sync, Double DQN online-selection/target-evaluation semantics, legal-action masking, checkpoint round-trip and no-memory transform. Mathematical tests use synthetic arrays strictly for utility checking.

Every neural training run has 120,000 transitions and 29,751 updates. Tabular runs have 120,000 actual TD updates each. Total training transitions 1,080,000; six neural run gradient updates 178,506. Main three DDQN elapsed 44.949 seconds; all nine logged training 111.384 seconds. Machine: Apple M5 Max arm64, CPU, one PyTorch thread. Timings exclude setup/evaluation/plotting. Versions: Python 3.12.5, NumPy 2.2.6, PyTorch 2.8.0, Matplotlib 3.9.2, pytest 9.0.3. Fresh virtual-environment dependency installation was not repeated.

Core source SHA-256 is `342709f0edec919c70b44d35d814ab56161f126926482ef0b35e21171220c40e`; top-level `src/` remains identical to the training source snapshot. Later changes affected docs/CLI help/tests/report/package tooling, without changing the algorithm/environment or retraining on test observations. All final checkpoints use the fixed budget; no best-seed selection or held-out tuning took place.

The report author/reviewer independently checked the raw means/primary paired intervals, compiled the PDF to exactly two pages and inspected rendered pages; see `report/VALIDATION.md` and `report/RESULTS_CROSSCHECK.json`. Packaging verifies CRC/path safety and every included SHA-256, while excluding `.git`, caches, virtual environments and duplicate `runs/` data.

Runtime checks establish internal consistency of this simulator experiment. They do not establish operational deployment effectiveness. The DQN did not outperform the alert rule control on the main comparison, and successful explicit-confirmation coverage was low; these remain prominent results.
