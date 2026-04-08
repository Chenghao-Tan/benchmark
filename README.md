# Benchmark Project

Run the example experiment:

```bash
python main.py -p experiment/toy/toydata_linear_toy.yaml
```

## Extended evaluation, results, and dashboard

The pipeline supports **per-instance metrics** (robustness, temporal decay, OpenXAI-style
`PGI_rec` / `PGU_rec` / `RIS_rec` / `ROS_rec`, plausibility KDE, action costs, fairness gaps,
runtime/memory) plus **JSON + Parquet** exports and a **Streamlit** dashboard.

- **Full experiment with exports** (optional Streamlit):

  ```bash
  python main.py -p experiment/toy/toydata_linear_toy_reproduce.yaml --eval-full --sample 10 --results-dir ./results/run1/ --dashboard
  ```

- **Sample script** (fast defaults, first *N* test rows):

  ```bash
  python run_sample_eval.py --config experiment/toy/toydata_linear_toy_reproduce.yaml --n_instances 10 --output results/sample/
  ```

Outputs:

- `results/.../raw/{dataset}_{model}_{method}_instances.json` — one dict per instance (gitignored under `raw/`).
- `results/.../summary/{dataset}_{model}_{method}.parquet` — aggregated mean/std/median/min/max + `Reliable_Recourse_Score`.

**Dashboard** (after a run produced files under `results/...`):

```bash
streamlit run visualization/dashboard.py -- --results_dir results/sample/
```

Optional YAML block `evaluation_full:` merges with defaults (see `evaluation/per_instance.default_evaluation_full_config`). Example keys: `robustness.n_trials`, `reliability.skip_ris`, `fairness.protected_feature`, `action_cost_weights`.
