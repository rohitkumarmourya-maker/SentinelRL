# Trained revised checkpoints

All checkpoints were produced by the frozen 2,500-episode protocol after the user authorized model execution. They use format version 2, 56 visible input features, 25 actions and real PyTorch autograd.

For each seed 17, 29, 43:

- `ddqn_history_seedS.pt`: primary final observation-history Double DQN.
- `ddqn_history_seedS_initial.pt`: its initial untrained network control.
- `ddqn_no_memory_seedS.pt`: separately retrained ablation, zeroing EMA/diagnosis/age features.
- `ddqn_no_memory_seedS_initial.pt`: that ablation's initial checkpoint, retained for provenance.
- `tabular_q_seedS.json`: genuine full-action tabular Q-learning baseline.

There are six trained neural networks, six initial neural checkpoints and three tabular checkpoints. Every trained DQN received 120,000 transitions and 29,751 gradient updates; all final parameter hashes differ from their initial hashes. The final checkpoints were fixed by budget, never selected by held-out score. Seed 17 is used for the documented demo by convention, not chosen as the best test seed.

`results/manifest_*.json` records exact source/checkpoint hashes, versions, elapsed training time and counts. The immutable source hash is `342709f0edec919c70b44d35d814ab56161f126926482ef0b35e21171220c40e`; source copies are in `provenance/training-source/`. Evaluation requires this source identity and the original environment configuration. Python/PyTorch version differences can affect exact numeric reproducibility.

Legacy checkpoints under `../legacy/models/` are an incompatible original system; they must not replace these files.
