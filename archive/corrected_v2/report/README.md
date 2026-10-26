# SentinelRL final empirical report

`main.pdf` is the two-page, two-column report, including references. `main.tex` and the vector figures in `assets/` are editable sources. The report documents completed fixed-budget experiments; it reports both the learned improvements and the weak explicit-confirmation behavior.

Build the PDF:

```sh
cd report
./build.sh
```

Tectonic must be on PATH and may download LaTeX packages on its first run. The vector figures are bundled, so building the PDF requires no model execution or plotting dependencies.

Regenerate figures from the project root with NumPy and Matplotlib installed:

```sh
python report/assets/make_figures.py
python report/assets/make_empirical_figures.py
```

The first command draws the fixed topology. The second defaults to the packaged `results/` directory, requires completed manifests and full 2,500-episode training logs, and reads the recorded CSVs only. It does not train, evaluate or load a model. An alternate completed result directory may be selected with `--results /absolute/path/to/results`.

Final rendered pages: `preview/page-1.png`, `preview/page-2.png`. Reference verification is in `REFERENCES_VERIFIED.md`; data checks are in `RESULTS_CROSSCHECK.json` and `VALIDATION.md`.
