"""
Streamlit dashboard: load results from --results_dir (raw JSON + summary Parquet).

Run: streamlit run visualization/dashboard.py -- --results_dir results/sample/
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
    list_instance_jsons,
    list_summary_parquets,
    load_all_summaries,
    load_instances,
    method_label_from_filename,
    parse_args_results_dir,
)


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

    labels = [method_label_from_filename(p) for p in json_paths]
    choice = st.sidebar.selectbox("Run / method bundle", list(zip(labels, json_paths)), format_func=lambda x: x[0])
    selected_path = choice[1]
    records = load_instances(selected_path)

    summary_all = load_all_summaries(pq_paths)
    stem = selected_path.stem.replace("_instances", "")
    summ_path = results_dir / "summary" / f"{stem}.parquet"
    summary_row = None
    if summ_path.is_file():
        summary_row = pd.read_parquet(summ_path).iloc[0]
    elif not summary_all.empty:
        summary_row = summary_all.iloc[0]

    st.title("Recourse reliability & trade-offs")

    tab_ov, tab_dist, tab_rel, tab_inst, tab_fair, tab_lb, tab_ex = st.tabs(
        [
            "Overview",
            "Distributions",
            "Reliability",
            "Instance browser",
            "Fairness",
            "Leaderboard",
            "Extras",
        ]
    )

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    with tab_ov:
        c1, c2 = st.columns(2)
        with c1:
            if summary_row is not None:
                fr = fig_radar(summary_row, title=f"Radar — {choice[0]}")
                st.plotly_chart(fr, use_container_width=True)
                save_fig_html(fr, out_dir / f"radar_{choice[0]}.html")
        with c2:
            fp = fig_pareto_3d(records, title="Pareto-style 3D view")
            st.plotly_chart(fp, use_container_width=True)
            save_fig_html(fp, out_dir / f"pareto_{choice[0]}.html")

    with tab_dist:
        metric = st.selectbox("Metric", ["cost_l1", "RIS_rec", "PGI_rec", "robustness_score", "runtime_seconds"])
        fv = fig_violin_metric(records, metric, choice[0])
        st.plotly_chart(fv, use_container_width=True)
        save_fig_html(fv, out_dir / f"violin_{metric}.html")

    with tab_rel:
        if summary_all.empty:
            st.info("No summary Parquet found.")
        else:
            st.plotly_chart(fig_reliability_heatmap(summary_all.head(1)), use_container_width=True)
        st.caption("Live perturbation simulator: use Instance browser + re-run pipeline for full fidelity.")

    with tab_inst:
        idx_pick = st.selectbox(
            "Instance",
            list(range(len(records))),
            format_func=lambda j: f"id {records[j].get('instance_id', j)}",
        )
        rec = records[idx_pick]
        st.plotly_chart(fig_parallel_instance(rec), use_container_width=True)
        fn = rec.get("feature_names") or []
        st.write("Badges (method-level):", difficulty_badges(summary_row) if summary_row is not None else [])

    with tab_fair:
        st.plotly_chart(fig_fairness_bars(summary_all.head(1)), use_container_width=True)

    with tab_lb:
        if summary_all.empty:
            st.warning("No Parquet summaries for leaderboard.")
        else:
            lb = build_leaderboard(summary_all)
            st.dataframe(lb, use_container_width=True)

    with tab_ex:
        st.subheader("Recourse Arena (pairwise win-rate + Elo proxy)")
        if len(pq_paths) >= 2 and not summary_all.empty:
            if "bundle_id" in summary_all.columns:
                names = summary_all["bundle_id"].astype(str).tolist()
            else:
                names = summary_all["source_file"].astype(str).tolist()
            scores = summary_all["Reliable_Recourse_Score"].astype(float).tolist()
            st.plotly_chart(fig_elo_heatmap(names, scores), use_container_width=True)
        else:
            st.info("Load multiple summary Parquet files for arena heatmap.")

        st.subheader("Sankey — feature-change paths")
        sk = fig_sankey_feature_changes(records)
        st.plotly_chart(sk, use_container_width=True)
        save_fig_html(sk, out_dir / f"sankey_{choice[0]}.html")

        st.subheader("UMAP manifold")
        dim = len(records[0].get("factual", [])) if records else 0
        fu = fig_umap_points(records, feature_dim=dim)
        st.plotly_chart(fu, use_container_width=True)
        save_fig_html(fu, out_dir / f"umap_{choice[0]}.html")

        if st.button("Export storytelling HTML"):
            path = out_dir / f"story_{choice[0]}.html"
            write_story_html(path, choice[0], [("Sankey", sk), ("UMAP", fu)])
            st.success(f"Wrote {path}")


if __name__ == "__main__":
    main()
