# Final empirical report verification

- `main.pdf` contains exactly **2 A4 pages including references**, with two columns, both requested team names/IDs and the exact repository hyperlink at the top.
- Tectonic compilation succeeded (exit 0). All citations resolve. No overfull boxes occur; one underfull-paragraph warning is harmless in the visually verified confirmation-delay definition.
- Both final pages rendered with Poppler at a 1600-pixel long edge and visually inspected after the final PDF rebuild. No clipped text, overlapping labels/equations, overflow or stale third page.
- Formulation cross-checked against the frozen implementation: augmented state S56 differs from policy observation56; actions25, budget6, horizon48; patch .20, decoy .25; complete reward including undetected-area penalty; exact DDQN target and hyperparameters.
- Completed source run: `runs/full_20261026`; canonical distributable evidence: project `results/`. All eleven directly used CSV/manifest inputs were confirmed byte-identical between these locations. Figure regeneration was tested using its packaged `results/` default, without running any model.
- Independently recomputed every primary summary mean from 2,100 primary episode records, confirmed unique policy/seed/scenario keys, and checked 6,300 auxiliary records. No recorded invalid actions or intervention-budget violations.
- Independently recomputed the reported crossed-bootstrap paired return differences against alert rule, tabular Q, random control, initialization and no-memory ablation, plus the critical-compromise ablation difference. All 2,000-resample bounds match generated result tables. Details: `RESULTS_CROSSCHECK.json`.
- Nine training manifests are COMPLETE with 120,000 transitions each. The main three DDQN runs each recorded 29,751 gradient updates and changed parameter hashes; their logged training wall times sum to 44.949 seconds. Runtime CPU/thread facts were supplied by the implementation agent's verified machine read.
- Empirical plots distinguish exploratory training curves from greedy held-out estimates. Comparison intervals use crossed seed/scenario bootstrap, not 300 independent training runs. Deterministic controls' repeated blocks add no independent scenarios.
- The report states the primary DDQN-versus-alert comparison is inconclusive and foregrounds low successful explicit confirmation. The separate sensor-sensitivity sample is not substituted for the primary test. No broad superiority or real-network effectiveness claim is made.
- Three standard algorithm/formulation references were verified against primary sources; see `REFERENCES_VERIFIED.md`.

Final PDF SHA-256: `287f66077c0d91e16511c6c08d2ae813493c79d9b8c331844abcde7283fd6de3`.

This report lane only plotted recorded results and compiled the document. Training, evaluation and model tests were run by the implementation agent after the user lifted the execution hold.
