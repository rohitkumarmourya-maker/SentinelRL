# Frozen selected models

`deployment.json` identifies the actual primary policy. It uniformly averages ten learned-Q networks, saved at 10k/20k from the five prespecified training seeds 17, 29, 43, 71, 89. This is one fixed deployment ensemble, not ten independent training runs. Every member has its actual seed/time, parameter hash and immutable source snapshot.

`runs/` contains 33 distinct selected neural/Q-table checkpoint files for the champion and retrained comparators. Previous corrected checkpoints remain in `archive/corrected_v2/models/`; strict original checkpoints remain in `vendor/original_v1/models/`. Their canonical hashes are bound by the sealed runtime. The original tabular historical/retrained tables are genuine persisted tables, not substitutes from the corrected implementation.

Use `python demo.py` and `python scripts/verify_champion.py --replay` from the project root. The public GitHub repository has not been modified.
