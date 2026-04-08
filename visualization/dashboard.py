"""
Streamlit dashboard: load results from --results_dir (raw JSON + summary Parquet).

Run: streamlit run visualization/dashboard.py -- --results_dir results/sample/

Performance: Streamlit reruns this file on every interaction. We cache file loads and
use sidebar navigation (not tabs) so heavy plots (UMAP, Sankey) run only when that
view is selected. Optional HTML export avoids disk writes on every rerun.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from visualization.extras import difficulty_badges, fig_elo_heatmap, write_story_html
from visualization.plots import (
    fig_fairness_bars,
    fig_parallel_instance,
    fig_pareto_3d,
    fig_radar,
    fig_reliability_heatmap,
    fig_sankey_feature_changes,
    fig_umap_points,
    fig_violin_metric,
    save_fig_html,
)
from visualization.utils_viz import (
    build_leaderboard,
    format_run_label,
    list_instance_jsons,
    list_summary_parquets,
    load_all_summaries,
    load_run_bundle,
    parse_args_results_dir,
)


@st.cache_data(show_spinner="Loading run...")
def _cached_bundle(json_path_str: str, mtime: float) -> dict:
    return load_run_bundle(Path(json_path_str))


@st.cache_data(show_spinner="Loading label...")
def _cached_run_label(json_path_str: str, mtime: float) -> str:
    b = load_run_bundle(Path(json_path_str))
    return format_run_label(Path(json_path_str), b.get("metadata"))


@st.cache_data(show_spinner="Loading summaries...")
def _cached_summaries(parquet_tuples: tuple[tuple[str, float], ...]) -> pd.DataFrame:
    paths = [Path(p) for p, _ in parquet_tuples]
    return load_all_summaries(paths)


def main() -> None:
    st.set_page_config(
        page_title="Recourse benchmark dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    results_dir = parse_args_results_dir()

    st.sidebar.header("Data")
    st.sidebar.write(f"**results_dir:** `{results_dir}`")

    json_paths = list_instance_jsons(results_dir)
    pq_paths = list_summary_parquets(results_dir)
    if not json_paths:
        st.error(f"No `*_instances.json` found under {results_dir / 'raw'}")
        st.stop()

    def _mtime(p: Path) -> float:
        try:
            return float(p.stat().st_mtime)
        except OSError:
            return 0.0

    selected_path = st.sidebar.selectbox(
        "Run (dataset | model | method)",
        json_paths,
        format_func=lambda p: _cached_run_label(str(p), _mtime(p)),
    )

    mt = _mtime(selected_path)
    bundle = _cached_bundle(str(selected_path), mt)
    records = bundle["instances"]
    run_metadata = bundle.get("metadata")

    pq_key = tuple((str(p), p.stat().st_mtime) for p in pq_paths)
    summary_all = _cached_summaries(pq_key) if pq_paths else pd.DataFrame()

    stem = selected_path.stem.replace("_instances", "")
    summ_path = results_dir / "summary" / f"{stem}.parquet"
    summary_row = None
    if summ_path.is_file():
        summary_row = pd.read_parquet(summ_path).iloc[0]
    elif not summary_all.empty:
        summary_row = summary_all.iloc[0]

    export_html = st.sidebar.checkbox(
        "Save Plotly HTML to visualization/outputs/",
        value=False,
        help="Off by default: writing files every rerun slows the app.",
    )
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tabs still execute all bodies on each rerun; use radio so only one view runs.
    page = st.sidebar.radio(
        "View",
        [
            "Overview",
            "Distributions",
            "Reliability",
            "Instance browser",
            "Fairness",
            "Leaderboard",
            "Extras",
        ],
        index=0,
    )

    st.title("Recourse reliability & trade-offs")

    with st.expander("**Current run: dataset / model / method**", expanded=True):
        if run_metadata:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Dataset", str(run_metadata.get("dataset", "—")))
            m2.metric("Model", str(run_metadata.get("model", "—")))
            m3.metric("Method", str(run_metadata.get("method", "—")))
            ex = str(run_metadata.get("experiment_name") or "").strip() or "—"
            m4.metric("Experiment (YAML `name`)", ex)
            st.caption(
                f"**File:** `{selected_path.name}`  |  **bundle_id:** `{run_metadata.get('bundle_id', stem)}`  |  "
                f"**n_instances:** {run_metadata.get('n_instances', len(records))}  |  "
                f"**device:** {run_metadata.get('device') or '—'}  |  "
                f"**sample cap:** {run_metadata.get('sample_instances') if run_metadata.get('sample_instances') is not None else 'full test'}"
            )
            snap = run_metadata.get("config_snapshot")
            if snap:
                st.markdown("**Config snapshot** (dataset / model / method blocks from YAML)")
                st.json(snap)
        else:
            st.warning(
                "This JSON has **no metadata block** (older export). "
                f"Filename stem **`{stem}`** is usually `{{dataset}}_{{model}}_{{method}}`."
            )

    if page == "Overview":
        c1, c2 = st.columns(2)
        with c1:
            if summary_row is not None:
                fr = fig_radar(summary_row, title=f"Radar — {stem}")
                st.plotly_chart(fr, width="stretch")
                if export_html:
                    save_fig_html(fr, out_dir / f"radar_{stem}.html")
        with c2:
            fp = fig_pareto_3d(records, title="Pareto-style 3D view")
            st.plotly_chart(fp, width="stretch")
            if export_html:
                save_fig_html(fp, out_dir / f"pareto_{stem}.html")

    elif page == "Distributions":
        metric = st.selectbox(
            "Metric",
            ["cost_l1", "RIS_rec", "PGI_rec", "robustness_score", "runtime_seconds"],
        )
        fv = fig_violin_metric(records, metric, stem)
        st.plotly_chart(fv, width="stretch")
        if export_html:
            save_fig_html(fv, out_dir / f"violin_{metric}.html")

    elif page == "Reliability":
        if summary_all.empty:
            st.info("No summary Parquet found.")
        else:
            heat_df = summary_all.head(1)
            if "bundle_id" in summary_all.columns:
                sub = summary_all[summary_all["bundle_id"].astype(str) == stem]
                if len(sub) >= 1:
                    heat_df = sub.iloc[[0]]
            fh = fig_reliability_heatmap(heat_df)
            st.plotly_chart(fh, width="stretch")
            if export_html:
                save_fig_html(fh, out_dir / f"reliability_{stem}.html")
        st.caption(
            "Live perturbation simulator: use Instance browser + re-run pipeline for full fidelity."
        )

    elif page == "Instance browser":
        idx_pick = st.selectbox(
            "Instance",
            list(range(len(records))),
            format_func=lambda j: f"id {records[j].get('instance_id', j)}",
        )
        rec = records[idx_pick]
        st.plotly_chart(fig_parallel_instance(rec), width="stretch")
        st.write(
            "Badges (method-level):",
            difficulty_badges(summary_row) if summary_row is not None else [],
        )

    elif page == "Fairness":
        fair_df = summary_all.head(1)
        if "bundle_id" in summary_all.columns:
            sub = summary_all[summary_all["bundle_id"].astype(str) == stem]
            if len(sub) >= 1:
                fair_df = sub.iloc[[0]]
        st.plotly_chart(fig_fairness_bars(fair_df), width="stretch")

    elif page == "Leaderboard":
        if summary_all.empty:
            st.warning("No Parquet summaries for leaderboard.")
        else:
            lb = build_leaderboard(summary_all)
            priority = [
                "dataset",
                "model",
                "method",
                "experiment_name",
                "bundle_id",
                "Reliable_Recourse_Score",
                "n_instances",
            ]
            front = [c for c in priority if c in lb.columns]
            rest = [c for c in lb.columns if c not in front]
            lb = lb[front + rest]
            st.dataframe(lb, width="stretch")

    elif page == "Extras":
        st.subheader("Recourse Arena (pairwise win-rate + Elo proxy)")
        if len(pq_paths) >= 2 and not summary_all.empty:
            if "bundle_id" in summary_all.columns:
                names = summary_all["bundle_id"].astype(str).tolist()
            else:
                names = summary_all["source_file"].astype(str).tolist()
            scores = summary_all["Reliable_Recourse_Score"].astype(float).tolist()
            fe = fig_elo_heatmap(names, scores)
            st.plotly_chart(fe, width="stretch")
            if export_html:
                save_fig_html(fe, out_dir / f"arena_{stem}.html")
        else:
            st.info("Load multiple summary Parquet files for arena heatmap.")

        st.subheader("Sankey — feature-change paths")
        sk = fig_sankey_feature_changes(records)
        st.plotly_chart(sk, width="stretch")
        if export_html:
            save_fig_html(sk, out_dir / f"sankey_{stem}.html")

        st.subheader("UMAP manifold")
        if len(records) < 40:
            st.info(
                "Only **"
                + str(len(records))
                + "** instances in this file — UMAP will look sparse. "
                "Re-run with a larger `--sample` (e.g. 150) on **German credit** "
                "(`experiment/german/german_linear_gs_umap_demo.yaml`) for a denser plot."
            )
        fu = fig_umap_points(records, feature_dim=None)
        st.plotly_chart(fu, width="stretch")
        if export_html:
            save_fig_html(fu, out_dir / f"umap_{stem}.html")

        if st.button("Export storytelling HTML"):
            path = out_dir / f"story_{stem}.html"
            write_story_html(path, stem, [("Sankey", sk), ("UMAP", fu)])
            st.success(f"Wrote {path}")

    st.sidebar.caption(
        "Tip: first load is slower; switching views avoids recomputing UMAP/Sankey."
    )


if __name__ == "__main__":
    main()
