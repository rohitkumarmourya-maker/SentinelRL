# Recovering source and evidence

The project preserves the original team's five authentic commits. The champion delivery adds one real, current snapshot commit after review; it does not manufacture an incremental historical narrative. No remote push was performed. The automated delivery commit uses an explicitly disclosed Codex identity rather than pretending to be either student.

To restore the complete history and current tracked tree:

```sh
git clone provenance/champion-history.bundle SentinelRL-restored
cd SentinelRL-restored
python scripts/reconstruct_results.py
```

The ZIP itself already contains the consolidated raw CSV. In Git, that 344 MB file is ignored; its lossless gzip is 16.7 MB and each immutable chunk is below the GitHub file limit. Reconstruction verifies the original SHA-256. GitHub blocks files larger than 100 MiB; see [GitHub's file-size documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). The bundle files are ZIP delivery artifacts, not recursively tracked blobs.

`provenance/original-history.bundle` independently retains the original history. `archive/corrected_v2/` contains the earlier corrected trained delivery; `historical/original_github_reproduction/` contains actual reproduced original benchmark evidence. Neither is substituted for the champion's sealed results. Model runtimes are imported from verified immutable snapshots under `models/runs/*/source/`, and selected file hashes are fixed in the protocol.
