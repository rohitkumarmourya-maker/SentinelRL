# LEGACY — incompatible original system, not corrected-version evidence

These are the user's supplied artifacts from commit `dff57ab019fde89506ea2969d7c384ce532a2bbf` (2026-10-24), preserved for provenance. The supplied ZIP and GitHub HEAD matched exactly when compared on 2026-10-26.

- `source/`: original Python modules, scripts, tests, docs, configuration and report.
- `models/`: three original checkpoints.
- `results/`: original logs/plots/claims.

**They do not validate the revised implementation.** The old observation indirectly exposed hidden infections and actual entry state; true detection counters leaked hidden information. Its learned Q output was clamped to ±0.12 before adding a much larger hand-coded action prior, and a further shield overrode choices. Therefore its reported results cannot establish genuine RL-driven performance or be transferred to the corrected environment. No attempt was made to run/validate these models during this delivery.

Original README claims and report figures inside this archive are historical source material, not current claims. Their earlier development dates/authors were not independently audited beyond preserving the repository's existing Git metadata. Use `provenance/original-history.bundle` for the original tree/history if needed. Never load these checkpoints with the revised DQN or report their numbers as corrected results.
