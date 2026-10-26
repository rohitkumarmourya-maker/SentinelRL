# Source and history provenance

The source is the user's own [SentinelRL repository](https://github.com/rohitkumarmourya-maker/SentinelRL), cloned without rewriting its history. The supplied WhatsApp ZIP was compared with that clone on 2026-10-26; `diff -rq` excluding `.git` found no differences.

- Original HEAD / ZIP git-archive comment: `dff57ab019fde89506ea2969d7c384ce532a2bbf`.
- Supplied ZIP SHA-256: `e34d81c0f4a30247284021b1104350fa1f6a3a4b50e15373038596c1259191a1`.
- Original file hashes: `provenance/original-file-hashes.json`.
- Machine-readable comparison: `provenance/source-comparison.json`.
- Unchanged original five-commit history: `provenance/original-history.bundle`.

Original commits, newest first:

| Commit | Recorded author date | Recorded author |
|---|---|---|
| `dff57ab` | 2026-10-24 | Rohit Mourya |
| `b58d40c` | 2026-10-22 | Rohit Mourya |
| `068868d` | 2026-10-19 | Rohit Mourya |
| `8db4e81` | 2026-10-16 | Jyotiraditya123467 |
| `b848ce7` | 2026-10-14 | rohitkumarmourya-maker |

These are preserved repository metadata, not independent proof of work times or original model correctness. No earlier commit dates/authors were invented or modified. The corrected delivery is a current working-tree revision; no retroactive development history or invented new author identity was created. No push or external message was sent.

To inspect the original snapshot separately:

```sh
git clone provenance/original-history.bundle SentinelRL-original
git -C SentinelRL-original log --oneline --all
```

That restores the historical original, including its known flaws; it does not restore the corrected working tree. The delivered current source is carried directly in the archive and covered by `provenance/delivery-file-hashes.json`. `legacy/README.md` explains why historical checkpoints/results cannot validate the revision.

Implementation reuses the team's source concept/topology and replaces flawed internals. No third-party project source or algorithm implementation was copied. PyTorch/NumPy/Matplotlib/pytest are declared external dependencies, not vendored source. The report cites primary algorithm/POMDP references separately.
