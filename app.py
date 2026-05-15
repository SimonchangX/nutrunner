"""
Atlas Copco ToolsNet8 - Trace Anomaly Detection Dashboard
Interactive web app for analyzing tightening curve data

Auto-retrain: On startup, checks for new/changed/removed data files.
If changes detected, automatically retrains the model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from anomaly_detector import TraceAnomalyDetector
from model_manager import (
    check_for_new_data,
    train_model,
    load_model,
    load_saved_metadata,
    get_folder_fingerprint,
    load_all_traces,
    MODEL_PATH,
)

st.set_page_config(
    page_title="ToolsNet8 Trace Anomaly Detection",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔧 Atlas Copco ToolsNet8 - Trace Anomaly Detection System")
st.markdown("**Detect abnormal tightening traces even when final values are within specifications**")


@st.cache_data(ttl=3600)
def load_all_data(folder_path, cache_buster=None):
    """Load all Excel files (uses model_manager cache for speed)"""
    combined_df, files = load_all_traces(folder_path, use_cache=True)
    return combined_df, files


def run_auto_retrain(folder_path, contamination):
    """Run retraining and store results in session state"""
    messages = []

    def capture_log(msg):
        messages.append(msg)

    result = train_model(folder_path, contamination=contamination, callback=capture_log)

    if result is None:
        st.session_state["retrain_status"] = "failed"
        st.session_state["retrain_messages"] = messages
        return

    detector, anomalies_df, info = result
    st.session_state["detector"] = detector
    st.session_state["anomalies_df"] = anomalies_df
    st.session_state["analysis_complete"] = True
    st.session_state["retrain_status"] = "success"
    st.session_state["retrain_info"] = info
    st.session_state["retrain_messages"] = messages
    st.session_state["last_train_fingerprint"] = info["fingerprint"]


def main():
    # Sidebar
    st.sidebar.header("📁 Data Configuration")
    folder_path = st.sidebar.text_input(
        "Data Folder Path",
        value=r"C:\Users\SZHH59\PyCharmMiscProject\Nutrunner\Anti-roll bar"
    )

    if not os.path.exists(folder_path):
        st.sidebar.error("Folder path does not exist!")
        st.stop()

    contamination = st.sidebar.slider(
        "Expected Anomaly Rate (%)",
        min_value=1, max_value=50, value=5, step=1,
        help="Expected percentage of anomalous traces"
    )

    # ====== CHECK: Has data changed since last training? ======
    has_changes, current_fp, saved_fp, new_files, removed_files = check_for_new_data(folder_path)
    saved_model = load_model()

    # ---- Scenario 1: No changes, saved model exists -> Load it directly ----
    if not has_changes and saved_model is not None:
        st.session_state["detector"] = saved_model["detector"]
        # Restore _features_df for PCA analysis
        if "_features_df" in saved_model and saved_model["_features_df"] is not None:
            st.session_state["detector"]._features_df = saved_model["_features_df"]
        # Restore calibrated rules
        if "calibrated_physics_rules" in saved_model:
            st.session_state["detector"].calibrated_physics_rules = saved_model["calibrated_physics_rules"]
        st.session_state["analysis_complete"] = True

        # Load saved results CSV
        results_path = os.path.join(folder_path, "anomaly_detection_results.csv")
        if os.path.exists(results_path):
            st.session_state["anomalies_df"] = pd.read_csv(results_path)

        meta = load_saved_metadata()
        st.sidebar.success(f"✅ Model loaded ({meta['file_count']} files, {meta['trace_count']} traces)")
        if meta:
            mtime = os.path.getmtime(MODEL_PATH)
            import time
            st.sidebar.caption(f"Last trained: {time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))}")

    # ---- Scenario 2: Data changed, saved model exists -> Offer to retrain ----
    elif has_changes and saved_model is not None:
        meta = load_saved_metadata()
        old_info = f"Previously: {meta.get('file_count', '?')} files, {meta.get('trace_count', '?')} traces"

        st.sidebar.warning("⚠️ Data files changed since last training!")
        if new_files:
            st.sidebar.success(f"📥 New files: {', '.join(new_files)}")
        if removed_files:
            st.sidebar.warning(f"🗑️ Removed: {', '.join(removed_files)}")
        st.sidebar.info(old_info)

        if st.sidebar.button("🔄 Retrain Now", type="primary"):
            run_auto_retrain(folder_path, contamination / 100)
            st.rerun()

        # Still load the old model so user can browse old results
        st.session_state["detector"] = saved_model["detector"]
        # Restore _features_df for PCA analysis
        if "_features_df" in saved_model and saved_model["_features_df"] is not None:
            st.session_state["detector"]._features_df = saved_model["_features_df"]
        if "calibrated_physics_rules" in saved_model:
            st.session_state["detector"].calibrated_physics_rules = saved_model["calibrated_physics_rules"]
        results_path = os.path.join(folder_path, "anomaly_detection_results.csv")
        if os.path.exists(results_path):
            st.session_state["anomalies_df"] = pd.read_csv(results_path)
        st.session_state["analysis_complete"] = True
        st.sidebar.caption("Showing previous results — retrain to include new data")

    # ---- Scenario 3: No saved model at all -> Need first training ----
    else:
        st.sidebar.info("📭 No trained model found")
        if st.sidebar.button("🚀 Train Model", type="primary"):
            run_auto_retrain(folder_path, contamination / 100)
            st.rerun()

    # Show retrain status banner (if just completed)
    if st.session_state.get("retrain_status") == "success":
        info = st.session_state.get("retrain_info", {})
        st.success(
            f"✅ **Model retrained!** {info.get('file_count', '?')} files, "
            f"{info.get('trace_count', '?')} traces, "
            f"{info.get('anomaly_count', '?')} anomalies detected"
        )
    elif st.session_state.get("retrain_status") == "failed":
        st.error("❌ Retraining failed. Check logs below:")
        for msg in st.session_state.get("retrain_messages", []):
            st.text(msg)

    # Load data with cache busting
    current_fp_val = get_folder_fingerprint(folder_path)[0]
    with st.spinner("Loading data from Excel files..."):
        df, files = load_all_data(folder_path, cache_buster=current_fp_val)

    if df is None:
        st.stop()

    st.sidebar.success(f"✅ Loaded {len(files)} files")
    st.sidebar.info(f"📊 {len(df):,} total samples")
    st.sidebar.info(f"🔩 {df['结果 ID'].nunique()} tightening results")

    # Initialize detector
    detector = TraceAnomalyDetector()
    detector.results_df = detector.extract_traces(df)

    # Use trained detector from session if available
    if st.session_state.get("analysis_complete") and "detector" in st.session_state:
        # Merge saved anomaly results with fresh raw trace data
        saved_anomalies = st.session_state.get("anomalies_df")
        if saved_anomalies is not None and "is_anomaly" in saved_anomalies.columns:
            # Only grab anomaly-specific columns that don't exist in results_df
            anomaly_cols = ["anomaly_score", "physics_score", "zscore_score", "ml_score",
                           "is_anomaly", "anomaly_reasons"]
            available_anomaly_cols = [c for c in anomaly_cols if c in saved_anomalies.columns]

            # Also grab source_file if available
            extra_cols = ["source_file"]
            available_extra = [c for c in extra_cols if c in saved_anomalies.columns and c not in detector.results_df.columns]

            merge_cols = ["result_id"] + available_anomaly_cols + available_extra
            merge_df = saved_anomalies[[c for c in merge_cols if c in saved_anomalies.columns]].copy()

            # Merge anomaly scores/reasons into fresh results_df which has raw traces
            merged = detector.results_df.merge(
                merge_df,
                on="result_id",
                how="left",
                suffixes=("", "_saved")
            )
            # Drop duplicate columns from suffixes
            dup_cols = [c for c in merged.columns if c.endswith("_saved")]
            merged = merged.drop(columns=dup_cols)

            # Drop traces that weren't in the training data (NaN anomaly_score)
            before = len(merged)
            merged = merged.dropna(subset=["anomaly_score"])
            if len(merged) < before:
                st.caption(f"⚠️ {before - len(merged)} trace(s) not in trained model — showing {len(merged)} of {before}")

            detector.anomalies_df = merged

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🔍 Anomaly Detection", "📈 Trace Viewer",
        "🏆 Model Performance", "🧬 PCA Analysis", "📋 Export Results"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Files", len(files))
        col2.metric("Total Samples", f"{len(df):,}")
        col3.metric("Tightening Results", df['结果 ID'].nunique())
        col4.metric("Program", df['程序'].unique()[0] if len(df['程序'].unique()) == 1 else "Multiple")
        
        st.subheader("Final Torque Distribution")
        fig_torque = go.Figure()
        fig_torque.add_trace(go.Histogram(
            x=detector.results_df['final_torque'],
            nbinsx=50,
            marker_color='steelblue'
        ))
        fig_torque.update_layout(
            xaxis_title="Final Torque (N·m)",
            yaxis_title="Count",
            height=300
        )
        st.plotly_chart(fig_torque, use_container_width=True, key="torque_dist")
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("Final Angle Distribution")
            fig_angle = go.Figure()
            fig_angle.add_trace(go.Histogram(
                x=detector.results_df['final_angle'],
                nbinsx=50,
                marker_color='coral'
            ))
            fig_angle.update_layout(
                xaxis_title="Final Angle (deg)",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig_angle, use_container_width=True, key="angle_dist")
        
        with col6:
            st.subheader("Samples per Result")
            fig_samples = go.Figure()
            fig_samples.add_trace(go.Histogram(
                x=detector.results_df['samples_count'],
                nbinsx=30,
                marker_color='seagreen'
            ))
            fig_samples.update_layout(
                xaxis_title="Sample Count",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig_samples, use_container_width=True, key="samples_dist")
    
    # Tab 2: Anomaly Detection
    with tab2:
        st.header("🔍 Anomaly Detection Results")

        if not st.session_state.get("analysis_complete"):
            st.info("Use **Train Model** or **Retrain Now** in the sidebar to run detection.")
        else:
            anomalies_df = detector.anomalies_df

            if anomalies_df is None or "is_anomaly" not in anomalies_df.columns:
                st.error("❌ Anomaly detection results are missing or corrupted. Please retrain the model.")
                st.stop()

            st.subheader("Detection Results")

            col1, col2, col3, col4 = st.columns(4)

            total = len(anomalies_df)
            anomaly_count = int(anomalies_df['is_anomaly'].fillna(False).sum())

            col1.metric("Total Traces", total)
            col2.metric("Normal Traces ✅", total - anomaly_count)
            col3.metric("Anomalous Traces ⚠️", int(anomaly_count))
            col4.metric("Anomaly Rate", f"{anomaly_count/total*100:.1f}%")

            # Show anomalous traces
            st.subheader("🚨 Detected Anomalies")
            is_anom = anomalies_df['is_anomaly'].fillna(False).astype(bool)
            anomalous = anomalies_df[is_anom]

            if len(anomalous) > 0:
                anomaly_display = anomalies_df[
                    is_anom
                ][[
                    'result_id', 'source_file', 'final_torque', 'final_angle',
                    'anomaly_score', 'physics_score', 'zscore_score', 'ml_score',
                    'anomaly_reasons'
                ]].copy()
                anomaly_display.columns = [
                    'Result ID', 'Source File', 'Final Torque', 'Final Angle',
                    'Total Score', 'Physics (40%)', 'Z-Score (20%)', 'ML (40%)',
                    'Reasons'
                ]
                st.dataframe(anomaly_display, use_container_width=True)

                # Score contribution visualization
                st.subheader("📊 Score Contributions")
                st.info("Each trace is scored by 3 layers. Bars show each layer's contribution (0 = normal, 1 = highly anomalous).")

                score_cols = ['Physics (40%)', 'Z-Score (20%)', 'ML (40%)', 'Total Score']
                score_data = anomaly_display[score_cols].copy()
                score_data.index = anomaly_display['Result ID']
                fig_scores = go.Figure()
                for col in score_cols:
                    fig_scores.add_trace(go.Bar(
                        name=col,
                        x=score_data.index,
                        y=score_data[col],
                        hovertemplate=f'<b>{col}</b><br>Score: %{{y:.3f}}<extra></extra>'
                    ))

                fig_scores.update_layout(
                    barmode='group',
                    xaxis_title="Result ID",
                    yaxis_title="Anomaly Score (0-1)",
                    height=400,
                    yaxis_range=[0, 1.05]
                )
                st.plotly_chart(fig_scores, use_container_width=True, key="anomaly_scores")
            else:
                st.info("No anomalies detected with current settings!")
    
    # Tab 3: Trace Viewer
    with tab3:
        st.header("📈 Trace Viewer")

        if st.session_state.get('analysis_complete', False):
            anomalies_df = detector.anomalies_df

            if anomalies_df is None or "is_anomaly" not in anomalies_df.columns:
                st.error("❌ Anomaly detection results are missing or corrupted. Please retrain the model.")
                st.stop()

            col1, col2 = st.columns(2)
            
            view_mode = col1.radio(
                "View Mode",
                ["Single Trace", "Compare Traces", "All Anomalies"]
            )
            
            if view_mode == "Single Trace":
                result_ids = anomalies_df['result_id'].tolist()
                selected_id = col2.selectbox("Select Result ID", result_ids)
                
                fig = detector.plot_single_trace_detail(selected_id)
                st.plotly_chart(fig, use_container_width=True, key="single_trace")
                
                # Show details
                trace_info = anomalies_df[anomalies_df['result_id'] == selected_id].iloc[0]
                st.subheader("Trace Information")
                st.write(f"**Result ID:** {selected_id}")
                st.write(f"**Status:** {'⚠️ ANOMALY' if trace_info.get('is_anomaly', False) else '✓ NORMAL'}")
                if pd.notna(trace_info.get('anomaly_reasons')):
                    st.write(f"**Reasons:** {trace_info['anomaly_reasons']}")
            
            elif view_mode == "Compare Traces":
                st.info("Select traces to compare")

                col3, col4 = st.columns(2)

                is_anom = anomalies_df['is_anomaly'].fillna(False).astype(bool)
                anomalous_ids = anomalies_df[is_anom]['result_id'].tolist()
                normal_ids = anomalies_df[~is_anom]['result_id'].tolist()
                
                selected_anomaly = col3.selectbox("Anomalous Trace", anomalous_ids[:10] if len(anomalous_ids) > 0 else [])
                selected_normal = col4.selectbox("Normal Trace", normal_ids[:10] if len(normal_ids) > 0 else [])
                
                if selected_anomaly and selected_normal:
                    compare_ids = [selected_anomaly, selected_normal]
                    fig = detector.plot_trace_comparison(compare_ids)
                    st.plotly_chart(fig, use_container_width=True, key="compare_traces")
            
            elif view_mode == "All Anomalies":
                is_anom = anomalies_df['is_anomaly'].fillna(False).astype(bool)
                anomalous_df = anomalies_df[is_anom].copy()
                anomalous_df = anomalous_df.sort_values('anomaly_score', ascending=False)
                all_anomalous_ids = anomalous_df['result_id'].tolist()
                
                st.info(f"📊 {len(all_anomalous_ids)} anomalous traces detected")
                
                # Show all in a table first
                st.dataframe(
                    anomalous_df[[
                        'result_id', 'source_file', 'final_torque', 'final_angle',
                        'anomaly_score', 'physics_score', 'zscore_score', 'ml_score',
                        'anomaly_reasons'
                    ]],
                    use_container_width=True,
                    height=300
                )
                
                # Pagination for trace plots (too many traces crash the browser)
                col5, col6 = st.columns(2)
                traces_per_page = col5.slider("Traces per page", 3, 20, 10, step=1)
                total_pages = max(1, (len(all_anomalous_ids) + traces_per_page - 1) // traces_per_page)
                page = col6.number_input("Page", 1, total_pages, 1)
                
                start_idx = (page - 1) * traces_per_page
                end_idx = min(start_idx + traces_per_page, len(all_anomalous_ids))
                page_ids = all_anomalous_ids[start_idx:end_idx]
                
                st.caption(f"Showing traces {start_idx + 1}-{end_idx} of {len(all_anomalous_ids)}")
                
                if page_ids:
                    fig = detector.plot_trace_comparison(page_ids)
                    st.plotly_chart(fig, use_container_width=True, key="all_anomalies")
        
        else:
            st.info("Please run anomaly detection first in the 'Anomaly Detection' tab!")

    # ====== Tab 4: Model Performance ======
    with tab4:
        st.header("🏆 Model Performance — Best vs Worst Traces")

        if not st.session_state.get("analysis_complete"):
            st.info("Run detection first to see model performance.")
        else:
            anomalies_df = detector.anomalies_df

            if anomalies_df is None or "anomaly_score" not in anomalies_df.columns:
                st.error("❌ Anomaly detection results are missing or corrupted. Please retrain the model.")
                st.stop()

            # Drop traces with no anomaly score (safety check)
            valid = anomalies_df.dropna(subset=["anomaly_score"])
            if len(valid) < len(anomalies_df):
                st.caption(f"⚠️ {len(anomalies_df) - len(valid)} trace(s) have no anomaly score — excluded from ranking")

            sorted_df = valid.sort_values("anomaly_score", ascending=True)

            # Top 10 best (lowest scores = most normal)
            best_10 = sorted_df.head(10)
            # Top 10 worst (highest scores = most anomalous)
            worst_10 = sorted_df.tail(10).iloc[::-1]

            # ---- Score distribution overview ----
            st.subheader("📊 Overall Score Distribution")

            col1, col2 = st.columns(2)

            with col1:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=anomalies_df["anomaly_score"],
                    nbinsx=50,
                    marker_color="steelblue"
                ))
                fig_hist.update_layout(
                    xaxis_title="Anomaly Score (0=normal, 1=highly anomalous)",
                    yaxis_title="Count",
                    height=280
                )
                st.plotly_chart(fig_hist, use_container_width=True, key="score_hist")

            with col2:
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(
                    y=anomalies_df["anomaly_score"],
                    name="All Traces",
                    boxpoints="outliers",
                    marker_color="steelblue"
                ))
                fig_box.add_trace(go.Box(
                    y=best_10["anomaly_score"],
                    name="Best 10",
                    boxpoints="all",
                    marker_color="green"
                ))
                fig_box.add_trace(go.Box(
                    y=worst_10["anomaly_score"],
                    name="Worst 10",
                    boxpoints="all",
                    marker_color="red"
                ))
                fig_box.update_layout(yaxis_title="Anomaly Score", height=280)
                st.plotly_chart(fig_box, use_container_width=True, key="score_box")

            # ---- Score breakdown by layer ----
            st.subheader("🔬 Score Breakdown: Physics vs Z-Score vs ML")

            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=anomalies_df["physics_score"],
                y=anomalies_df["ml_score"],
                mode="markers",
                marker=dict(
                    size=6,
                    color=anomalies_df["anomaly_score"],
                    colorscale="RdYlBu_r",
                    showscale=True,
                    colorbar=dict(title="Total Score")
                ),
                text=[f"ID: {r}<br>Source: {s}<br>Torque: {t:.1f} Nm"
                      for r, s, t in zip(
                          anomalies_df["result_id"],
                          anomalies_df.get("source_file", [""] * len(anomalies_df)),
                          anomalies_df["final_torque"]
                      )],
                hovertemplate="<b>%{text}</b><br>"
                              "Physics: %{x:.3f}<br>"
                              "ML: %{y:.3f}<extra></extra>",
                name="all"
            ))
            fig_scatter.update_layout(
                xaxis_title="Physics Score (40%)",
                yaxis_title="ML Score (40%)",
                height=400
            )
            st.plotly_chart(fig_scatter, use_container_width=True, key="score_scatter")

            # ---- Best 10 Traces ----
            st.subheader("✅ 10 Most Normal Traces (Lowest Anomaly Scores)")

            best_display = best_10[[
                "result_id", "source_file", "final_torque", "final_angle",
                "anomaly_score", "physics_score", "zscore_score", "ml_score"
            ]].copy()
            best_display.columns = [
                "Result ID", "Source File", "Final Torque (Nm)", "Final Angle (°)",
                "Total Score", "Physics (40%)", "Z-Score (20%)", "ML (40%)"
            ]
            st.dataframe(best_display, use_container_width=True, height=280)

            # ---- Worst 10 Traces ----
            st.subheader("⚠️ 10 Most Anomalous Traces (Highest Anomaly Scores)")

            worst_display = worst_10[[
                "result_id", "source_file", "final_torque", "final_angle",
                "anomaly_score", "physics_score", "zscore_score", "ml_score",
                "anomaly_reasons"
            ]].copy()
            worst_display.columns = [
                "Result ID", "Source File", "Final Torque (Nm)", "Final Angle (°)",
                "Total Score", "Physics (40%)", "Z-Score (20%)", "ML (40%)",
                "Reasons"
            ]
            st.dataframe(worst_display, use_container_width=True, height=280)

            # ---- Best vs Worst comparison plots ----
            st.subheader("📈 Best vs Worst — Torque & Stiffness Comparison")

            best_ids = best_10["result_id"].tolist()
            worst_ids = worst_10["result_id"].tolist()

            tab_best, tab_worst, tab_compare = st.tabs([
                "✅ Best 10 Traces", "⚠️ Worst 10 Traces", "🔄 Side-by-Side"
            ])

            with tab_best:
                fig_best = detector.plot_trace_comparison(best_ids)
                st.plotly_chart(fig_best, use_container_width=True, key="best_traces")

            with tab_worst:
                fig_worst = detector.plot_trace_comparison(worst_ids)
                st.plotly_chart(fig_worst, use_container_width=True, key="worst_traces")

            with tab_compare:
                st.info("Each pair shows 1 best (green) + 1 worst (red) trace overlaid.")

                for i in range(10):
                    best_id = best_ids[i]
                    worst_id = worst_ids[i]
                    st.caption(
                        f"Pair {i+1}: Best ID {best_id} (score: {best_10.iloc[i]['anomaly_score']:.3f})  vs  "
                        f"Worst ID {worst_id} (score: {worst_10.iloc[i]['anomaly_score']:.3f})"
                    )

                    # Build custom comparison plot for this pair
                    pair_ids = [best_id, worst_id]
                    traces_dict = {row["result_id"]: row for _, row in detector.anomalies_df.iterrows()}

                    fig_pair = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=("Torque vs Time", "Stiffness (dTorque/dAngle)"),
                        vertical_spacing=0.12, shared_xaxes=True
                    )

                    for tid in pair_ids:
                        if tid not in traces_dict:
                            continue
                        td = traces_dict[tid]
                        is_best = tid == best_id
                        color = "green" if is_best else "red"
                        label = f"Best ID {tid}" if is_best else f"Worst ID {tid}"

                        fig_pair.add_trace(go.Scatter(
                            x=td["time_series"], y=td["torque_time_series"],
                            mode="lines", name=f"{label} — Torque",
                            line=dict(width=2, color=color)
                        ), row=1, col=1)

                        stiffness = detector.calculate_stiffness(
                            td["torque_time_series"], td["angle_time_series"]
                        )
                        t_stiff = td["time_series"][:-1] if len(td["time_series"]) > 1 else td["time_series"]

                        fig_pair.add_trace(go.Scatter(
                            x=t_stiff, y=stiffness,
                            mode="lines", name=f"{label} — Stiffness",
                            line=dict(width=2, color=color),
                            showlegend=False
                        ), row=2, col=1)

                    fig_pair.update_layout(height=500)
                    fig_pair.update_xaxes(title_text="Time (ms)", row=2, col=1)
                    fig_pair.update_yaxes(title_text="Torque (N·m)", row=1, col=1)
                    fig_pair.update_yaxes(title_text="Stiffness (N·m/deg)", row=2, col=1)

                    st.plotly_chart(fig_pair, use_container_width=True, key=f"pair_{i}")



        if st.session_state.get('analysis_complete', False):
            detector = st.session_state["detector"]

            if hasattr(detector, 'plot_pca_anomalies'):
                fig_pca = detector.plot_pca_anomalies()
                if fig_pca:
                    st.plotly_chart(fig_pca, use_container_width=True, key="pca_chart")

            st.subheader("Calibrated Physics Thresholds")
            if hasattr(detector, 'calibrated_physics_rules') and detector.calibrated_physics_rules:
                rules = detector.calibrated_physics_rules
                cols = st.columns(3)
                cols[0].metric("Stiffness Drop", f"< {rules.get('stiffness_drop_thresh', 0):.1f}")
                cols[1].metric("Overshoot", f"> {rules.get('overshoot_thresh', 0):.3f}")
                cols[2].metric("Linearity", f"< {rules.get('linearity_thresh', 0):.3f}")
                cols = st.columns(3)
                cols[0].metric("Flat Regions", f"> {rules.get('flat_regions_thresh', 0):.2f}")
                cols[1].metric("High Freq Energy", f"> {rules.get('high_freq_thresh', 0):.3f}")
                cols[2].metric("Stiffness Ratio", f"> {rules.get('stiffness_ratio_thresh', 0):.0f}")
            else:
                st.info("Run training first to see calibrated thresholds.")
        else:
            st.info("Run detection first to see PCA analysis!")

    # Tab 6: Export Results
    with tab6:
        st.header("📋 Export Results")

        if st.session_state.get('analysis_complete', False):
            anomalies_df = detector.anomalies_df

            if anomalies_df is None or "is_anomaly" not in anomalies_df.columns:
                st.error("❌ Anomaly detection results are missing or corrupted. Please retrain the model.")
                st.stop()

            st.subheader("Download Results")
            
            # Prepare export data
            export_df = anomalies_df[[
                'result_id', 'source_file', 'final_torque', 'final_angle',
                'max_torque', 'max_angle', 'is_anomaly', 'anomaly_score',
                'physics_score', 'zscore_score', 'ml_score', 'anomaly_reasons'
            ]].copy()
            export_df.columns = [
                'Result ID', 'Source File', 'Final Torque (N·m)', 'Final Angle (deg)',
                'Max Torque (N·m)', 'Max Angle (deg)', 'Is Anomaly', 'Total Anomaly Score',
                'Physics Score (40%)', 'Z-Score (20%)', 'ML Score (40%)', 'Anomaly Reasons'
            ]
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="anomaly_detection_results.csv",
                mime="text/csv"
            )
            
            # Summary report
            st.subheader("Summary Report")
            total = len(export_df)
            anomalies = export_df['Is Anomaly'].sum()
            
            st.write(f"**Total Traces Analyzed:** {total}")
            st.write(f"**Normal Traces:** {total - anomalies} ({(total-anomalies)/total*100:.1f}%)")
            st.write(f"**Anomalous Traces:** {anomalies} ({anomalies/total*100:.1f}%)")
            
            if anomalies > 0:
                st.warning(f"⚠️ {anomalies} traces detected as anomalous! Review the 'Trace Viewer' tab for details.")
            else:
                st.success("✅ All traces appear normal!")
        
        else:
            st.info("Please run anomaly detection first!")


if __name__ == "__main__":
    main()
