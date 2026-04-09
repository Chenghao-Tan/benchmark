# Benchmark Live Site

Static dashboard for method ranking from benchmark outputs.

## Features
- Choose `dataset`, `model`, and optional subset of `methods`.
- Select ranking metrics with weights and optimization direction.
- Add per-metric min/max constraints.
- Upload a YAML experiment config to auto-apply dataset/model/evaluation metric choices.
- View top methods and full ranking table.

## Data Source
The dashboard reads:

- `data/benchmark_results.csv`

Current file is generated from the 2x2x5 benchmark batch.

## Run Locally
Use any static server from repo root, for example:

```powershell
python -m http.server 8080
```

Then open:

- `http://localhost:8080/webapp/benchmark-live/`

## GitHub Pages
Yes, this can be hosted on GitHub Pages because it is static HTML/CSS/JS.

Limitations on GitHub Pages:
- It can rank and filter existing benchmark results.
- It cannot execute new Python experiments server-side.
