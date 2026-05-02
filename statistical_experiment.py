import copy
import csv
import json
import os
import random
import shutil
import statistics
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

import genetic_optimizer as go
import surrogate_model as sm


BASE_CACHE_FILE = Path("airfoil_cache.json")
OUTPUT_DIR = Path(".statistical_experiment")
SUMMARY_FILE = OUTPUT_DIR / "statistical_summary.txt"
RESULTS_FILE = OUTPUT_DIR / "statistical_results.json"
TABLE_FILE = OUTPUT_DIR / "statistical_results.csv"
PLOT_FILE = OUTPUT_DIR / "statistical_convergence.png"

CONFIGS = [
    {"name": "GA", "use_ml": False, "use_rl": False},
    {"name": "GA + ML", "use_ml": True, "use_rl": False},
    {"name": "GA + ML + RL", "use_ml": True, "use_rl": True},
]

SEEDS = [42, 52, 62, 72, 82]


def _ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def _slugify(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
    )


def _prepare_cache_file(path, source_file=None):
    if source_file is None:
        path.write_text("{}\n", encoding="utf-8")
        return path

    if source_file.exists():
        shutil.copy2(source_file, path)
    else:
        path.write_text("{}\n", encoding="utf-8")
    return path


def _run_single(config, seed):
    run_slug = _slugify(config["name"])
    runtime_cache = OUTPUT_DIR / f"{run_slug}_seed_{seed}_runtime_cache.json"
    training_cache = OUTPUT_DIR / f"{run_slug}_seed_{seed}_training_cache.json"
    control_file = OUTPUT_DIR / f"{run_slug}_seed_{seed}_control.json"
    vis_file = OUTPUT_DIR / f"{run_slug}_seed_{seed}_visualization.json"

    _prepare_cache_file(runtime_cache)
    _prepare_cache_file(training_cache, source_file=BASE_CACHE_FILE)
    control_file.write_text(
        json.dumps({"running": True, "mission": dict(go.MISSION)}, indent=2),
        encoding="utf-8",
    )

    original_go_cache = go.CACHE_FILE
    original_sm_cache = sm.CACHE_FILE
    original_control_file = go.CONTROL_FILE
    original_vis_file = go.VIS_STATE_FILE
    original_fitness_cache = copy.deepcopy(go.fitness_cache)
    original_use_surrogate = go.USE_SURROGATE
    original_guided_search = go.ML_GUIDED_SEARCH
    original_rl_enabled = go.RL_ENABLED
    original_fixed_airfoil = go.FIXED_AIRFOIL
    original_mission = copy.deepcopy(go.MISSION)
    original_weight = go.WEIGHT

    try:
        random.seed(seed)
        np.random.seed(seed)

        go.CACHE_FILE = runtime_cache
        sm.CACHE_FILE = training_cache
        go.CONTROL_FILE = control_file
        go.VIS_STATE_FILE = vis_file
        go.fitness_cache = go.load_fitness_cache()

        result = go.run_ga(
            use_surrogate=config["use_ml"],
            guided_search=config["use_ml"],
            rl_enabled=config["use_rl"],
            seed=seed,
            return_metrics=True,
        )
        result["name"] = config["name"]
        result["seed"] = seed
        return result
    finally:
        go.CACHE_FILE = original_go_cache
        sm.CACHE_FILE = original_sm_cache
        go.CONTROL_FILE = original_control_file
        go.VIS_STATE_FILE = original_vis_file
        go.fitness_cache = original_fitness_cache
        go.USE_SURROGATE = original_use_surrogate
        go.ML_GUIDED_SEARCH = original_guided_search
        go.RL_ENABLED = original_rl_enabled
        go.FIXED_AIRFOIL = original_fixed_airfoil
        go.MISSION.clear()
        go.MISSION.update(original_mission)
        go.WEIGHT = original_weight


def _mean(values):
    return statistics.mean(values) if values else float("nan")


def _variance(values):
    return statistics.variance(values) if len(values) > 1 else 0.0


def _stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _aggregate_runs(results):
    grouped = {}
    for result in results:
        grouped.setdefault(result["name"], []).append(result)

    aggregated = []
    for config in CONFIGS:
        name = config["name"]
        runs = grouped.get(name, [])
        fitness = [run["best"]["mission_fitness"] for run in runs]
        xfoil_calls = [run["xfoil_calls"] for run in runs]
        runtime_seconds = [run["runtime_seconds"] for run in runs]
        ld = [run["best"]["raw_score"] for run in runs]

        aggregated.append(
            {
                "name": name,
                "runs": len(runs),
                "seeds": [run["seed"] for run in runs],
                "fitness_mean": _mean(fitness),
                "fitness_std": _stdev(fitness),
                "fitness_var": _variance(fitness),
                "ld_mean": _mean(ld),
                "ld_std": _stdev(ld),
                "ld_var": _variance(ld),
                "xfoil_mean": _mean(xfoil_calls),
                "xfoil_std": _stdev(xfoil_calls),
                "xfoil_var": _variance(xfoil_calls),
                "runtime_mean": _mean(runtime_seconds),
                "runtime_std": _stdev(runtime_seconds),
                "runtime_var": _variance(runtime_seconds),
                "ml_reduction_mean": _mean([run["xfoil_reduction_pct"] for run in runs]),
                "ml_reduction_std": _stdev([run["xfoil_reduction_pct"] for run in runs]),
                "rl_reward_mean": _mean([run["rl_average_reward"] for run in runs]),
                "rl_reward_std": _stdev([run["rl_average_reward"] for run in runs]),
            }
        )

    return aggregated


def _save_results(results, aggregated):
    payload = {"raw_runs": results, "aggregated": aggregated, "seeds": SEEDS}
    RESULTS_FILE.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _save_table(aggregated):
    rows = [
        [
            "Method",
            "Runs",
            "Seeds",
            "Fitness Mean",
            "Fitness Std",
            "Fitness Var",
            "L/D Mean",
            "L/D Std",
            "XFOIL Mean",
            "XFOIL Std",
            "Runtime Mean",
            "Runtime Std",
        ]
    ]

    for row in aggregated:
        rows.append(
            [
                row["name"],
                str(row["runs"]),
                ";".join(map(str, row["seeds"])),
                f"{row['fitness_mean']:.6f}",
                f"{row['fitness_std']:.6f}",
                f"{row['fitness_var']:.6f}",
                f"{row['ld_mean']:.6f}",
                f"{row['ld_std']:.6f}",
                f"{row['xfoil_mean']:.2f}",
                f"{row['xfoil_std']:.2f}",
                f"{row['runtime_mean']:.2f}",
                f"{row['runtime_std']:.2f}",
            ]
        )

    with TABLE_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _save_summary(aggregated):
    lines = [
        "STATISTICAL EXPERIMENT SUMMARY",
        "=" * 32,
        f"Seeds: {', '.join(map(str, SEEDS))}",
        "",
    ]

    for row in aggregated:
        lines.extend(
            [
                row["name"],
                f"  Runs: {row['runs']}",
                f"  Seeds: {', '.join(map(str, row['seeds']))}",
                f"  Fitness mean: {row['fitness_mean']:.6f}",
                f"  Fitness std: {row['fitness_std']:.6f}",
                f"  Fitness variance: {row['fitness_var']:.6f}",
                f"  L/D mean: {row['ld_mean']:.6f}",
                f"  L/D std: {row['ld_std']:.6f}",
                f"  XFOIL mean: {row['xfoil_mean']:.2f}",
                f"  XFOIL std: {row['xfoil_std']:.2f}",
                f"  Runtime mean (s): {row['runtime_mean']:.2f}",
                f"  Runtime std (s): {row['runtime_std']:.2f}",
                "",
            ]
        )

    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")


def _save_plot(aggregated):
    labels = [row["name"] for row in aggregated]
    fitness_means = [row["fitness_mean"] for row in aggregated]
    fitness_stds = [row["fitness_std"] for row in aggregated]
    runtime_means = [row["runtime_mean"] for row in aggregated]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x - width / 2, fitness_means, width=width, yerr=fitness_stds, capsize=5, label="Fitness", color="#2f9e44")
    ax1.set_ylabel("Mission Fitness")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(x + width / 2, runtime_means, marker="o", linewidth=2, color="#1c7ed6", label="Runtime")
    ax2.set_ylabel("Runtime (s)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc="upper left", bbox_to_anchor=(0.08, 0.92))
    fig.suptitle("Statistically Validated GA Comparison")
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_experiment():
    _ensure_output_dir()

    results = []
    for config in CONFIGS:
        print(f"\nRunning {config['name']}...")
        for seed in SEEDS:
            print(f"  Seed {seed}")
            start = time.perf_counter()
            result = _run_single(config, seed)
            elapsed = time.perf_counter() - start
            print(
                f"    fitness={result['best']['mission_fitness']:.4f} "
                f"xfoil={result['xfoil_calls']} "
                f"runtime={elapsed:.2f}s"
            )
            result["wall_time_seconds"] = elapsed
            results.append(result)

    aggregated = _aggregate_runs(results)
    _save_results(results, aggregated)
    _save_table(aggregated)
    _save_summary(aggregated)
    _save_plot(aggregated)

    print("\n=== Statistical Results ===")
    for row in aggregated:
        print(
            f"{row['name']}: "
            f"fitness={row['fitness_mean']:.4f}+/-{row['fitness_std']:.4f}, "
            f"xfoil={row['xfoil_mean']:.2f}+/-{row['xfoil_std']:.2f}, "
            f"runtime={row['runtime_mean']:.2f}+/-{row['runtime_std']:.2f}s"
        )

    print(f"\nSaved results to {RESULTS_FILE}")
    print(f"Saved table to {TABLE_FILE}")
    print(f"Saved summary to {SUMMARY_FILE}")
    print(f"Saved plot to {PLOT_FILE}")

    return {"raw_runs": results, "aggregated": aggregated}


if __name__ == "__main__":
    run_experiment()
