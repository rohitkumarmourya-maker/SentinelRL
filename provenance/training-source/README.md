# Immutable training source snapshot

This exact `src/` produced every revised bundled checkpoint and raw experiment table. Its combined SHA-256 is recorded in `source-sha256.txt` and in checkpoint/manifests. Top-level `src/` remains byte-identical. Documentation, CLI help, tests and report were subsequently updated without changing the trained algorithm.

The protocol's `FROZEN_FUTURE_PROTOCOL` status string is the preserved plan snapshot; each run's outer `execution_status: COMPLETE` and primary `results/STATUS.json` record actual execution after authorization.
