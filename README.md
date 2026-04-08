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

- `results/.../raw/{dataset}_{model}_{method}_instances.json` — wrapped as `{"metadata": {...}, "instances": [...]}`. Metadata includes **dataset**, **model**, **method**, **experiment_name** (YAML top-level `name`), **device**, **n_instances**, **sample_instances**, and a **config_snapshot** of the `dataset` / `model` / `method` YAML blocks. Legacy flat lists are still loaded by the dashboard.
- `results/.../summary/{dataset}_{model}_{method}.parquet` — same identifiers as columns plus mean/std/median/min/max and `Reliable_Recourse_Score`.

**Compare several methods** into one folder (one JSON+Parquet per method; dashboard lists them all):

```bash
# Recommended: preset with defaults for toy + linear (toy, wachter, dice, gs, claproar)
python run_compare_methods.py --config experiment/toy/toydata_linear_compare.yaml --preset toy_linear --output results/compare_toy/ --n_instances 10
```

```bash
# Custom list (YAML `method:` hyperparameters shared where possible; only `name` is swapped)
python run_compare_methods.py --config experiment/toy/toydata_linear_toy_reproduce.yaml --methods toy,wachter,dice --output results/compare/ --n_instances 10
```

Use a YAML that is valid for every method you list (shared preprocess/model). With `--preset toy_linear`, keep `method` minimal (`seed`, `device`, `desired_class`); per-method kwargs are filled in by `run_compare_methods.py`.

**Dashboard** (after a run produced files under `results/...`):

```bash
streamlit run visualization/dashboard.py -- --results_dir results/sample/
```

The sidebar shows each run as **`dataset | model | method (experiment_name) n=…`**. The top expander repeats the same fields and the YAML snapshot.

**Richer UMAP (more points):** German Credit is shipped with `german.csv` (~1k rows). Example with 150 test rows (~300 UMAP points after stacking factual+CF):

```bash
python main.py -p experiment/german/german_linear_gs_umap_demo.yaml --eval-full --sample 150 --results-dir results/german_umap/
streamlit run visualization/dashboard.py -- --results_dir results/german_umap/
```

Then open **View → Extras → UMAP manifold**.

Optional YAML block `evaluation_full:` merges with defaults (see `evaluation/per_instance.default_evaluation_full_config`). Example keys: `robustness.n_trials`, `reliability.skip_ris`, `fairness.protected_feature`, `action_cost_weights`.
