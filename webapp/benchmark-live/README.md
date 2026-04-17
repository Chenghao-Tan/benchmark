# Benchmark Live Site (CSV-Only)

This dashboard ranks recourse methods from precomputed experiment rows in:

- `data/benchmark_results.csv`

The page does not run experiments. It only filters and ranks rows already in the CSV.

## Run Locally
From repo root:

```powershell
venv12\Scripts\python.exe -m http.server 8000 --directory webapp\benchmark-live
```

Open:

- `http://127.0.0.1:8000/`

## Expected CSV Columns
Required filter columns:
- `dataset`
- `model`
- `method`

Metrics used by ranking:
- `validity` (maximize)
- `distance_l1` (minimize)
- `distance_l0` (minimize)
- `ynn` (maximize)
- `runtime_seconds` (minimize)

Optional:
- `status` (if present, only rows with `success` are used)
