# Final report verification

- Final PDF is exactly **2 A4 pages including references**, in two columns. Requested team names/IDs and GitHub hyperlink are at the top.
- Tectonic compilation succeeds (exit 0); all references resolve and there are no overfull boxes. Two modest underfull-paragraph warnings in the observation-feature description were visually reviewed and cause no clipping or overlap.
- Both final pages rendered with Poppler at a 1600-pixel long edge and were visually inspected after the last PDF rebuild. Figures, labels, equations, table and references are legible at the final column size.
- Frozen environment and method were statically cross-checked: reward-relevant state 56, public observation 56, exact 64-state causal posterior and 126-slot representation; exact inspection/transition/alarm ordering; shared-host dueling architecture, MSE, 3-step gamma 1 targets, public forced-WAIT compression and ten-checkpoint mean-Q deployment from five training runs.
- Streamed **520,000** final raw records independently in the report lane: 52 distinct policy/seed rows each cover exactly 10,000 unique scenarios `900000000+103*i`. No recorded invalid actions or budget violations. Recomputed 342 displayed-relevant summary means and all 55 primary paired return mean differences from raw records, including fixed baseline-checkpoint rows. All match at absolute tolerance1e-9.
- Actual intervals are bound directly to the completed `paired_comparisons.csv` and `summary_intervals.csv`; asymmetric familywise bounds are plotted without replacing them with symmetric errors. The frozen procedure shares scenario draws and resamples actual unequal comparator-seed axes, with 50,000 primary replicates and Bonferroni alpha.05/55; fixed ensembles have one policy row. Secondary 2,000-resample intervals remain exploratory.
- Result interpretation: 54 positive simultaneous return bounds; tuned-rule exception delta−.08051369 with adjusted interval[−.22305042,.06317902]. The report explicitly rejects universal superiority and equivalence claims. Confirmation coverage 8.62 percent is the mean episode fraction, not a pooled rate; compromise/isolation metrics remain counts per episode.
- Five final training logs each contain 20,000 episodes. Training evidence records 4.8 million actual environment steps and 1,719,035 optimizer updates with changed parameter hashes. Logged training elapsed times sum 3806.92177 worker-seconds; this is presented as 63.4 worker-minutes, not parallel wall-clock time.
- No models were executed by the report lane. Training/evaluation were run by the implementation/learner lanes under the authorized frozen protocol.

Frozen protocol SHA-256: `ad11c25a5389477c4351c11047b89fb48aca57f81b732b30795a77646bbccecf`.

Final PDF SHA-256: `236a685df60c3698baae1c639894af369b1980d507508b6b57b98dd2366475c5`.
