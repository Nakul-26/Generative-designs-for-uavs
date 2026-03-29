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
    page_title="UAV Optimizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(15, 118, 110, 0.22), transparent 28%),
            radial-gradient(circle at top right, rgba(37, 99, 235, 0.18), transparent 30%),
            linear-gradient(180deg, #0e1117 0%, #111827 100%);
        color: #e5e7eb;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1.5rem;
    }
    .stMetric {
        background-color: #1c1f26;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.18);
    }
    .dashboard-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(17, 24, 39, 0.96));
        color: white;
        border-radius: 22px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 20px 45px rgba(2, 6, 23, 0.38);
        margin-bottom: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.12);
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
        return {"running": True}

    try:
        with open(CONTROL_FILE, "r") as handle:
            control = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"running": True}

    return {"running": bool(control.get("running", True))}


def write_control(running):
    tmp_path = CONTROL_FILE.with_suffix(".json.tmp")

    with open(tmp_path, "w") as handle:
        json.dump({"running": running}, handle, indent=2)

    tmp_path.replace(CONTROL_FILE)


def build_population_frame(state):
    population = state.get("population", [])
    if not population:
        return pd.DataFrame(
            columns=[
                "airfoil",
                "label",
                "wing_span",
                "wing_area",
                "ld",
                "adjusted_fitness",
                "lift",
                "drag",
                "lift_margin",
                "constraint_satisfied",
                "aspect_ratio",
                "evaluation_type",
                "surrogate_mean_ld",
                "surrogate_uncertainty",
            ]
        )

    frame = pd.DataFrame(population)
    frame["evaluation_type"] = frame["evaluation_type"].fillna("unknown")
    if "label" not in frame.columns:
        frame["label"] = frame["airfoil"]
    return frame


def render_airfoil_plot(naca):
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    if not naca:
        ax.text(0.5, 0.5, "Waiting for optimizer...", ha="center", va="center", color="#e5e7eb")
        ax.axis("off")
        return fig

    xu, yu, xl, yl = generate_naca4_coordinates(naca)
    ax.plot(xu, yu, color="#f8fafc", linewidth=2.2)
    ax.plot(xl, yl, color="#f8fafc", linewidth=2.2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18, color="#475569")
    ax.set_xlabel("x/c", color="#cbd5e1")
    ax.set_ylabel("y/c", color="#cbd5e1")
    ax.set_title(naca, color="#f8fafc")
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#475569")
    return fig


def render_evaluation_mix_plot(source_counts):
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    labels = [label for label in EVALUATION_ORDER if source_counts.get(label, 0) > 0]

    if not labels:
        ax.text(0.5, 0.5, "No evaluation data yet", ha="center", va="center", color="#e5e7eb")
        ax.axis("off")
        return fig

    counts = [source_counts[label] for label in labels]
    colors = [EVALUATION_COLORS[label] for label in labels]
    ax.barh(labels, counts, color=colors)
    ax.set_xlabel("Count", color="#cbd5e1")
    ax.set_ylabel("Evaluation Type", color="#cbd5e1")
    ax.grid(True, axis="x", alpha=0.18, color="#475569")
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#475569")
    return fig


def render_population_performance_chart(chart_frame, best_label):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    if chart_frame.empty:
        ax.text(0.5, 0.5, "No population data yet", ha="center", va="center", color="#e5e7eb")
        ax.axis("off")
        return fig

    labels = chart_frame["label"].tolist()
    lds = chart_frame["ld"].tolist()
    colors = [
        "#00ff88" if label == best_label else "#444c5c"
        for label in labels
    ]

    ax.bar(range(len(lds)), lds, color=colors)
    ax.set_title("Population Performance", color="#f8fafc")
    ax.set_ylabel("L/D", color="#cbd5e1")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color="#cbd5e1")
    ax.tick_params(axis="y", colors="#cbd5e1")
    ax.grid(True, axis="y", alpha=0.18, color="#475569")
    for spine in ax.spines.values():
        spine.set_color("#475569")

    for i, label in enumerate(labels):
        if label == best_label:
            ax.text(i, lds[i], "*", ha="center", va="bottom", color="#facc15", fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig


def render_dashboard(
    state,
    population_frame,
    max_rows,
    selected_types,
    sort_column,
    show_surrogate_columns,
    explain_mode,
    beginner_mode,
):
    status = state.get("status", "waiting")
    generation = state.get("generation")
    total_generations = state.get("generations_total", "-")
    best_airfoil = state.get("best_airfoil")
    best_span = state.get("best_span")
    best_area = state.get("best_area")
    best_lift = state.get("best_lift")
    best_ld = state.get("best_ld")
    best_adjusted_fitness = state.get("best_adjusted_fitness")
    weight_target = state.get("weight_target")
    dynamic_pressure = state.get("dynamic_pressure")
    source_counts = state.get("source_counts", {})
    best_label = (
        f"{best_airfoil} | b={best_span:.2f}m | S={best_area:.2f}m^2"
        if best_airfoil and best_span is not None and best_area is not None
        else best_airfoil
    )

    st.markdown(
        f"""
        <div class="dashboard-header">
            <h1>UAV Wing Optimization Dashboard</h1>
            <p>Status: {status} | Generation: {generation if generation is not None else "-"} / {total_generations}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Generation", generation if generation is not None else "-")
    col2.metric("Best L/D", f"{best_ld:.2f}" if best_ld is not None else "-")
    col3.metric("Best Airfoil", best_airfoil or "-")
    col4.metric("Best Span (m)", f"{best_span:.2f}" if best_span is not None else "-")
    col5.metric("Best Area (m^2)", f"{best_area:.2f}" if best_area is not None else "-")

    if beginner_mode:
        st.success("System is automatically designing better wings.")

    if explain_mode:
        st.info(
            "Generation = current optimization step. "
            "Best L/D = best efficiency found so far. "
            "Best Airfoil = the current wing section shape. "
            "Span and Area = the current wing size."
        )

    if not beginner_mode:
        stat1, stat2, stat3, stat4 = st.columns(4)
        stat1.metric(
            "Fitness Score",
            f"{best_adjusted_fitness:.2f}" if best_adjusted_fitness is not None else "-",
        )
        stat2.metric("Best Lift", f"{best_lift:.2f}" if best_lift is not None else "-")
        stat3.metric("Real Simulations", state.get("xfoil_calls", 0))
        stat4.metric("AI Predictions", state.get("ml_predictions", 0))

        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        stats_col1.metric("Designs This Generation", len(population_frame))
        stats_col2.metric(
            "Run Time (s)",
            f"{state.get('runtime_seconds', 0):.1f}" if state.get("runtime_seconds") else "-",
        )
        stats_col3.metric("Status", status.title())
        stats_col4.metric("Early Rejects", state.get("ml_skips", 0))

        if explain_mode:
            st.info(
                "Fitness Score = ranking score used by the optimizer. "
                "Best Lift = how much upward force the best wing produces. "
                "Real Simulations = expensive physics runs. "
                "AI Predictions = model guesses used to speed things up. "
                "Early Rejects = weak designs skipped before full simulation."
            )

    st.markdown("")
    left, right = st.columns([2, 1])

    with left:
        st.subheader("How Good Are The Current Wings?")
        if explain_mode:
            st.info(
                "Each bar is one wing design in the current generation. "
                "Taller means better L/D. "
                "Green marks the best design."
            )
        filtered = population_frame[population_frame["evaluation_type"].isin(selected_types)].copy()
        if sort_column in filtered.columns:
            ascending = sort_column == "airfoil"
            filtered = filtered.sort_values(sort_column, ascending=ascending, kind="stable")

        chart_frame = filtered.head(max_rows)
        if chart_frame.empty:
            st.info("No population rows match the current filters.")
        else:
            performance_fig = render_population_performance_chart(chart_frame, best_label)
            st.pyplot(performance_fig, width="stretch")
            plt.close(performance_fig)

        if not beginner_mode:
            st.subheader("Detailed Population Data")
            display_columns = [
                "label",
                "ld",
                "wing_span",
                "wing_area",
                "lift",
                "drag",
                "lift_margin",
                "adjusted_fitness",
                "evaluation_type",
            ]
            if show_surrogate_columns:
                display_columns.extend(["surrogate_mean_ld", "surrogate_uncertainty"])
            st.dataframe(
                chart_frame[display_columns],
                width="stretch",
                hide_index=True,
            )

            if show_surrogate_columns:
                surrogate_columns = [
                    column
                    for column in ["label", "surrogate_mean_ld", "surrogate_uncertainty"]
                    if column in chart_frame.columns
                ]
                if len(surrogate_columns) > 1:
                    st.subheader("AI / Surrogate Details")
                    st.dataframe(
                        chart_frame[surrogate_columns],
                        width="stretch",
                        hide_index=True,
                    )

    with right:
        st.subheader("What Does The Best Wing Look Like?")
        if explain_mode:
            st.info(
                "This is the shape of the current best airfoil. "
                "The top and bottom curves are the two wing surfaces. "
                "Span and area are optimized separately from this 2D shape."
            )
        fig = render_airfoil_plot(best_airfoil)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        st.subheader("How Were Designs Evaluated?")
        if explain_mode:
            st.info(
                "This shows how the system judged designs: "
                "real simulation or cache reuse for the airfoil aerodynamics."
            )
        if not any(source_counts.get(label, 0) > 0 for label in EVALUATION_ORDER):
            st.info("Evaluation breakdown will appear after the first generation.")
        else:
            mix_fig = render_evaluation_mix_plot(source_counts)
            st.pyplot(mix_fig, width="stretch")
            plt.close(mix_fig)

    if beginner_mode:
        st.subheader("Is The System Improving?")
        if explain_mode:
            st.info(
                "This line shows the best L/D found over time. "
                "If it rises, the optimizer is finding better wing designs."
            )
        history = state.get("best_history", [])
        if history:
            history_frame = pd.DataFrame(
                {"generation": list(range(len(history))), "best_ld": history}
            ).set_index("generation")
            st.line_chart(history_frame, color="#2563eb")
        else:
            st.info("Convergence data will appear once the first generation completes.")
    else:
        history_col, details_col = st.columns([1.2, 0.8])
        with history_col:
            st.subheader("Is The System Improving?")
            if explain_mode:
                st.info(
                "This line shows the best L/D found over time. "
                "If it rises, the optimizer is finding better wing designs."
                )
            history = state.get("best_history", [])
            if history:
                history_frame = pd.DataFrame(
                    {"generation": list(range(len(history))), "best_ld": history}
                ).set_index("generation")
                st.line_chart(history_frame, color="#2563eb")
            else:
                st.info("Convergence data will appear once the first generation completes.")

        with details_col:
            st.subheader("What Happened This Step?")
            if explain_mode:
                st.info(
                "These counters summarize only the current generation: "
                    "full simulations and AI predictions for the airfoil analysis."
                )
            counter_cols = st.columns(3)
            counter_cols[0].metric("Simulated", state.get("generation_xfoil_calls", 0))
            counter_cols[1].metric("Predicted", state.get("generation_predictions", 0))
            counter_cols[2].metric("Skipped", state.get("generation_ml_skips", 0))

        st.subheader("Advanced Metrics")
        st.json(
            {
                "status": status,
                "generation": generation,
                "generations_total": total_generations,
                "best_airfoil": best_airfoil,
                "best_span": best_span,
                "best_area": best_area,
                "best_lift": best_lift,
                "weight_target": weight_target,
                "dynamic_pressure": dynamic_pressure,
                "best_ld": best_ld,
                "best_adjusted_fitness": best_adjusted_fitness,
                "xfoil_calls": state.get("xfoil_calls", 0),
                "ml_predictions": state.get("ml_predictions", 0),
                "ml_skips": state.get("ml_skips", 0),
                "generation_xfoil_calls": state.get("generation_xfoil_calls", 0),
                "generation_predictions": state.get("generation_predictions", 0),
                "generation_ml_skips": state.get("generation_ml_skips", 0),
                "source_counts": source_counts,
            },
            expanded=False,
        )


def main():
    st.sidebar.header("Controls")
    control = load_control()
    running = control["running"]

    start_col, stop_col = st.sidebar.columns(2)
    if start_col.button("Start", width="stretch", disabled=running):
        write_control(True)
        st.rerun()
    if stop_col.button("Stop", width="stretch", disabled=not running):
        write_control(False)
        st.rerun()

    st.sidebar.caption(f"Optimizer: {'Running' if running else 'Stopped'}")
    beginner_mode = st.sidebar.toggle("Beginner Mode", value=True)
    explain_mode = st.sidebar.toggle("Explain Mode", value=True)
    refresh_label = st.sidebar.selectbox(
        "Refresh interval",
        options=list(REFRESH_OPTIONS_MS.keys()),
        index=0,
    )
    if beginner_mode:
        selected_types = ["simulated", "cached", "skipped", "unknown"]
        sort_column = "ld"
        max_rows = 12
        show_surrogate_columns = False
    else:
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
        explain_mode=explain_mode,
        beginner_mode=beginner_mode,
    )

    if refresh_ms > 0:
        time.sleep(refresh_ms / 1000)
        st.rerun()


if __name__ == "__main__":
    main()
