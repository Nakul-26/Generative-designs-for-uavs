import json
import os
from pathlib import Path

_MPL_CONFIG_DIR = Path(".matplotlib")
_MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR.resolve()))

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from airfoil_geometry import generate_naca4_coordinates


STATE_FILE = Path("visualization_state.json")
COLORS = {
    "simulated": "#2a9d8f",
    "cached": "#577590",
    "skipped": "#e76f51",
    "unknown": "#9aa0a6",
}


def load_state():
    if not STATE_FILE.exists():
        return None

    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def plot_best_airfoil(ax, naca):
    ax.clear()
    ax.set_title("Best Airfoil Shape")

    if not naca:
        ax.text(0.5, 0.5, "Waiting for optimizer...", ha="center", va="center")
        ax.axis("off")
        return

    xu, yu, xl, yl = generate_naca4_coordinates(naca)
    ax.plot(xu, yu, color="#264653", linewidth=2)
    ax.plot(xl, yl, color="#264653", linewidth=2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(f"Best Airfoil Shape: {naca}")


def plot_population(ax, state):
    ax.clear()
    population = state.get("population", [])

    if not population:
        ax.text(0.5, 0.5, "No population data yet", ha="center", va="center")
        ax.set_axis_off()
        return

    lds = [entry["ld"] for entry in population]
    colors = [COLORS.get(entry.get("evaluation_type", "unknown"), COLORS["unknown"]) for entry in population]

    ax.bar(range(len(population)), lds, color=colors)
    ax.set_title("Population L/D")
    ax.set_xlabel("Population Index")
    ax.set_ylabel("L/D")
    ax.grid(True, axis="y", alpha=0.3)


def plot_history(ax, state):
    ax.clear()
    history = state.get("best_history", [])

    if not history:
        ax.text(0.5, 0.5, "Waiting for convergence data", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.plot(range(len(history)), history, marker="o", color="#1d3557")
    ax.set_title("Best L/D by Generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best L/D")
    ax.grid(True, alpha=0.3)


def plot_metrics(ax, state):
    ax.clear()
    ax.axis("off")

    source_counts = state.get("source_counts", {})
    lines = [
        f"Status: {state.get('status', 'unknown')}",
        f"Generation: {state.get('generation', '-')}/{state.get('generations_total', '-')}",
        f"Best airfoil: {state.get('best_airfoil', '-')}",
        f"Best L/D: {state.get('best_ld', 0):.2f}" if state.get("best_ld") is not None else "Best L/D: -",
        f"XFOIL calls: {state.get('xfoil_calls', 0)}",
        f"ML predictions: {state.get('ml_predictions', 0)}",
        f"ML skips: {state.get('ml_skips', 0)}",
        f"This gen simulated: {state.get('generation_xfoil_calls', 0)}",
        f"This gen predicted: {state.get('generation_predictions', 0)}",
        f"This gen skipped: {state.get('generation_ml_skips', 0)}",
        f"Population simulated: {source_counts.get('simulated', 0)}",
        f"Population cached: {source_counts.get('cached', 0)}",
        f"Population skipped: {source_counts.get('skipped', 0)}",
    ]

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=10,
    )


def update(_frame):
    state = load_state()

    if state is None:
        metrics_ax.clear()
        metrics_ax.axis("off")
        metrics_ax.text(0.5, 0.5, f"Waiting for {STATE_FILE.name}", ha="center", va="center")
        population_ax.clear()
        population_ax.axis("off")
        history_ax.clear()
        history_ax.axis("off")
        airfoil_ax.clear()
        airfoil_ax.axis("off")
        return

    generation = state.get("generation", "-")
    best_ld = state.get("best_ld")
    title_suffix = f"Generation {generation}"
    if best_ld is not None:
        title_suffix += f" | Best L/D {best_ld:.2f}"
    fig.suptitle(title_suffix, fontsize=14)

    plot_population(population_ax, state)
    plot_history(history_ax, state)
    plot_metrics(metrics_ax, state)
    plot_best_airfoil(airfoil_ax, state.get("best_airfoil"))


fig, ((population_ax, history_ax), (airfoil_ax, metrics_ax)) = plt.subplots(2, 2, figsize=(14, 8))
fig.tight_layout(pad=3.0)
animation = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)


if __name__ == "__main__":
    plt.show()
