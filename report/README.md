# SentinelRL champion report

`main.pdf` is the final two-page, two-column report, including references. Editable sources are `main.tex`, `verified_results.tex`, and the vector figures in `assets/`.

The report states the actual mixed outcome:54 of 55 adjusted return contrasts favor the fixed temporal ensemble; the tuned diagnostic rule has the better point estimate and its paired difference is inconclusive. It explicitly reports weak successful confirmation and does not claim universal or all-metric superiority.

## Build

From the project root:

```sh
cd report
./build.sh
```

Tectonic must be on PATH. Bundled vector figures make compilation independent of Python and model execution. Tectonic may download LaTeX packages on first use.

## Regenerate evidence figures and bound numerical text

From the project root, with NumPy and Matplotlib installed:

```sh
python report/assets/bind_results.py
python report/assets/make_empirical_figures.py
cd report
./build.sh
```

These scripts only read completed recorded CSV/JSON evidence; they do not load, train or evaluate models. `assets/figure_sources.json` binds the packaged `results/final_benchmark/` data and five canonical training logs. The report binds this frozen experiment, not an arbitrary new run. `--training-only` redraws training curves without reading final outcomes.

Final page renders are `preview/page-1.png` and `preview/page-2.png`. Primary-source references are documented in `REFERENCES_VERIFIED.md`; independent data checks appear in `RESULTS_CROSSCHECK.json` and `VALIDATION.md`.
