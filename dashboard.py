import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from airfoil_geometry import generate_naca4_coordinates


STATE_FILE = Path("visualization_state.json")
CONTROL_FILE = Path("control.json")
REFRESH_OPTIONS_MS = {
    "1 second": 1000,
    "2 seconds": 2000,
    "5 seconds": 5000,
    "Manual only": 0,
}
EVALUATION_ORDER = ["simulated", "cached", "skipped", "unknown"]
EVALUATION_COLORS = {
    "simulated": "#0f766e",
    "cached": "#2563eb",
    "skipped": "#dc2626",
    "unknown": "#6b7280",
}


st.set_page_config(
    page_title="UAV Airfoil Optimization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(37, 99, 235, 0.10), transparent 30%),
            linear-gradient(180deg, #f7fafc 0%, #eef4f7 100%);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    .dashboard-header {
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.94), rgba(37, 99, 235, 0.92));
        color: white;
        border-radius: 22px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.14);
        margin-bottom: 1rem;
    }
    .dashboard-header h1 {
        margin: 0;
        font-size: 2rem;
        line-height: 1.2;
    }
    .dashboard-header p {
        margin: 0.35rem 0 0;
        color: rgba(255, 255, 255, 0.88);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_state():
    if not STATE_FILE.exists():
        return None

    try:
        with open(STATE_FILE, "r") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def load_control():
    if not CONTROL_FILE.exists():
        return {"paused": False}

    try:
        with open(CONTROL_FILE, "r") as handle:
            control = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"paused": False}

    return {"paused": bool(control.get("paused", False))}


def write_control(paused):
    tmp_path = CONTROL_FILE.with_suffix(".json.tmp")

    with open(tmp_path, "w") as handle:
        json.dump({"paused": paused}, handle, indent=2)

    tmp_path.replace(CONTROL_FILE)


def build_population_frame(state):
    population = state.get("population", [])
    if not population:
        return pd.DataFrame(
            columns=[
                "airfoil",
                "ld",
                "adjusted_fitness",
                "evaluation_type",
                "surrogate_mean_ld",
                "surrogate_uncertainty",
            ]
        )

    frame = pd.DataFrame(population)
    frame["evaluation_type"] = frame["evaluation_type"].fillna("unknown")
    return frame


def render_airfoil_plot(naca):
    fig, ax = plt.subplots(figsize=(6, 3))

    if not naca:
        ax.text(0.5, 0.5, "Waiting for optimizer...", ha="center", va="center")
        ax.axis("off")
        return fig

    xu, yu, xl, yl = generate_naca4_coordinates(naca)
    ax.plot(xu, yu, color="#0f172a", linewidth=2.2)
    ax.plot(xl, yl, color="#0f172a", linewidth=2.2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(naca)
    return fig


def render_evaluation_mix_plot(source_counts):
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = [label for label in EVALUATION_ORDER if source_counts.get(label, 0) > 0]

    if not labels:
        ax.text(0.5, 0.5, "No evaluation data yet", ha="center", va="center")
        ax.axis("off")
        return fig

    counts = [source_counts[label] for label in labels]
    colors = [EVALUATION_COLORS[label] for label in labels]
    ax.barh(labels, counts, color=colors)
    ax.set_xlabel("Count")
    ax.set_ylabel("Evaluation Type")
    ax.grid(True, axis="x", alpha=0.25)
    return fig


def render_dashboard(state, population_frame, max_rows, selected_types, sort_column, show_surrogate_columns):
    status = state.get("status", "waiting")
    generation = state.get("generation")
    total_generations = state.get("generations_total", "-")
    best_airfoil = state.get("best_airfoil")
    best_ld = state.get("best_ld")
    best_adjusted_fitness = state.get("best_adjusted_fitness")
    source_counts = state.get("source_counts", {})

    st.markdown(
        f"""
        <div class="dashboard-header">
            <h1>UAV Airfoil Optimization Dashboard</h1>
            <p>Status: {status} | Generation: {generation if generation is not None else "-"} / {total_generations}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best L/D", f"{best_ld:.2f}" if best_ld is not None else "-")
    col2.metric("Best Airfoil", best_airfoil or "-")
    col3.metric(
        "Adjusted Fitness",
        f"{best_adjusted_fitness:.2f}" if best_adjusted_fitness is not None else "-",
    )
    col4.metric("Population Size", len(population_frame))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("XFOIL Calls", state.get("xfoil_calls", 0))
    col6.metric("ML Predictions", state.get("ml_predictions", 0))
    col7.metric("ML Skips", state.get("ml_skips", 0))
    col8.metric("Runtime (s)", f"{state.get('runtime_seconds', 0):.1f}" if state.get("runtime_seconds") else "-")

    st.markdown("")
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Population Performance")
        filtered = population_frame[population_frame["evaluation_type"].isin(selected_types)].copy()
        if sort_column in filtered.columns:
            ascending = sort_column == "airfoil"
            filtered = filtered.sort_values(sort_column, ascending=ascending, kind="stable")

        chart_frame = filtered.head(max_rows)
        if chart_frame.empty:
            st.info("No population rows match the current filters.")
        else:
            st.bar_chart(
                chart_frame.set_index("airfoil")["ld"],
                color="#0f766e",
                horizontal=True,
            )

        display_columns = ["airfoil", "ld", "adjusted_fitness", "evaluation_type"]
        if show_surrogate_columns:
            display_columns.extend(["surrogate_mean_ld", "surrogate_uncertainty"])
        st.dataframe(
            chart_frame[display_columns],
            width="stretch",
            hide_index=True,
        )

    with right:
        st.subheader("Best Airfoil Shape")
        fig = render_airfoil_plot(best_airfoil)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        st.subheader("Evaluation Mix")
        if not any(source_counts.get(label, 0) > 0 for label in EVALUATION_ORDER):
            st.info("Evaluation breakdown will appear after the first generation.")
        else:
            mix_fig = render_evaluation_mix_plot(source_counts)
            st.pyplot(mix_fig, width="stretch")
            plt.close(mix_fig)

    history_col, details_col = st.columns([1.2, 0.8])
    with history_col:
        st.subheader("Convergence")
        history = state.get("best_history", [])
        if history:
            history_frame = pd.DataFrame(
                {"generation": list(range(len(history))), "best_ld": history}
            ).set_index("generation")
            st.line_chart(history_frame, color="#2563eb")
        else:
            st.info("Convergence data will appear once the first generation completes.")

    with details_col:
        st.subheader("Generation Counters")
        counter_cols = st.columns(3)
        counter_cols[0].metric("Simulated", state.get("generation_xfoil_calls", 0))
        counter_cols[1].metric("Predicted", state.get("generation_predictions", 0))
        counter_cols[2].metric("Skipped", state.get("generation_ml_skips", 0))


def main():
    st.sidebar.header("Controls")
    control = load_control()
    paused = control["paused"]

    pause_col, resume_col = st.sidebar.columns(2)
    if pause_col.button("Pause", width="stretch", disabled=paused):
        write_control(True)
        st.rerun()
    if resume_col.button("Resume", width="stretch", disabled=not paused):
        write_control(False)
        st.rerun()

    st.sidebar.caption(f"Optimizer: {'Paused' if paused else 'Running'}")
    refresh_label = st.sidebar.selectbox(
        "Refresh interval",
        options=list(REFRESH_OPTIONS_MS.keys()),
        index=0,
    )
    selected_types = st.sidebar.multiselect(
        "Evaluation types",
        options=EVALUATION_ORDER,
        default=["simulated", "cached", "skipped"],
    )
    sort_column = st.sidebar.selectbox(
        "Sort population by",
        options=["ld", "adjusted_fitness", "airfoil"],
        index=0,
    )
    max_rows = st.sidebar.slider("Rows to display", min_value=5, max_value=30, value=12)
    show_surrogate_columns = st.sidebar.checkbox("Show surrogate columns", value=False)
    manual_refresh = st.sidebar.button("Refresh now", width="stretch")

    refresh_ms = REFRESH_OPTIONS_MS[refresh_label]
    if manual_refresh:
        st.rerun()

    state = load_state()
    if state is None:
        st.markdown(
            """
            <div class="dashboard-header">
                <h1>UAV Airfoil Optimization Dashboard</h1>
                <p>Waiting for visualization_state.json from the optimizer.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("Start the optimizer with `python main.py`, then refresh or enable auto-refresh.")
        return

    population_frame = build_population_frame(state)
    if not selected_types:
        selected_types = EVALUATION_ORDER

    render_dashboard(
        state=state,
        population_frame=population_frame,
        max_rows=max_rows,
        selected_types=selected_types,
        sort_column=sort_column,
        show_surrogate_columns=show_surrogate_columns,
    )

    if refresh_ms > 0:
        time.sleep(refresh_ms / 1000)
        st.rerun()


if __name__ == "__main__":
    main()
