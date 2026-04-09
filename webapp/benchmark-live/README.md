# Benchmark Live Site

This app has two modes:

1. Static mode (GitHub Pages): rank/filter existing CSV results.
2. Server mode (Python backend): run new experiments from UI and append fresh results.

## Files
- `index.html`, `styles.css`, `app.js`: frontend
- `server.py`: backend API + static file server
- `data/benchmark_results.csv`: baseline results
- `data/runtime_results.csv`: generated in server mode

## Server Mode (Recommended)
From repo root:

```powershell
venv12\Scripts\python.exe webapp\benchmark-live\server.py --port 8000
```

Open:

- `http://127.0.0.1:8000/`

API endpoints used by UI:
- `GET /api/options`
- `GET /api/results`
- `POST /api/run-selection`
- `POST /api/run-config`
- `GET /api/jobs/<job_id>`

## Static Mode (GitHub Pages)
GitHub Pages can host the dashboard UI, but it cannot execute Python experiments.

In static mode the page falls back to `data/benchmark_results.csv` only.
