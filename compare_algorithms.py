import copy
import json
import os
import random
import shutil
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

import genetic_optimizer as go
import surrogate_model as sm


BASE_CACHE_FILE = Path("airfoil_cache.json")
SUMMARY_FILE = Path("three_way_experiment_summary.txt")
PLOT_FILE = Path("three_way_convergence.png")
EXPERIMENT_DIR = Path(".three_way_experiment")


def _prepare_isolated_cache():
    temp_dir = EXPERIMENT_DIR
    temp_dir.mkdir(exist_ok=True)
    temp_cache = temp_dir / "airfoil_cache.json"
    if BASE_CACHE_FILE.exists():
        base_data = json.loads(BASE_CACHE_FILE.read_text(encoding="utf-8"))
        stripped = {}
        for airfoil, entry in base_data.items():
            if isinstance(entry, dict):
                stripped[airfoil] = {"ld": entry.get("ld", 0)}
            else:
                stripped[airfoil] = {"ld": entry}
        temp_cache.write_text(json.dumps(stripped, indent=2), encoding="utf-8")
    else:
        temp_cache.write_text("{}\n", encoding="utf-8")
    return temp_dir, temp_cache


def _run_mode(name, use_surrogate, guided_search, seed):
    temp_dir, temp_cache = _prepare_isolated_cache()
    original_cache_file = go.CACHE_FILE
    original_surrogate_cache_file = sm.CACHE_FILE
    original_fitness_cache = copy.deepcopy(go.fitness_cache)
    original_use_surrogate = go.USE_SURROGATE
    original_guided_search = go.ML_GUIDED_SEARCH

    try:
        random.seed(seed)
        go.CACHE_FILE = temp_cache
        go.fitness_cache = go.load_fitness_cache()
        sm.CACHE_FILE = BASE_CACHE_FILE
        go.USE_SURROGATE = use_surrogate
        go.ML_GUIDED_SEARCH = guided_search

        start = time.perf_counter()
        best, history = go.run_ga()
        runtime = time.perf_counter() - start

        guided_attempts = go.ml_guided_attempts
        guided_accepts = go.ml_guided_accepts

        return {
            "name": name,
            "best": best,
            "history": history,
            "runtime_s": runtime,
            "xfoil_calls": go.xfoil_calls,
            "ml_predictions": go.ml_predictions,
            "ml_skips": go.ml_skips,
            "guided_attempts": guided_attempts,
            "guided_accepts": guided_accepts,
            "guided_acceptance_rate": (
                (guided_accepts / guided_attempts) * 100.0 if guided_attempts else 0.0
            ),
            "cache_size": len(go.fitness_cache),
        }
    finally:
        go.CACHE_FILE = original_cache_file
        sm.CACHE_FILE = original_surrogate_cache_file
        go.fitness_cache = original_fitness_cache
        go.USE_SURROGATE = original_use_surrogate
        go.ML_GUIDED_SEARCH = original_guided_search
        shutil.rmtree(temp_dir, ignore_errors=True)


def _convergence_metrics(history):
    if not history:
        return {"best_generation": None, "gen_95pct": None}

    best_value = max(history)
    best_generation = history.index(best_value) + 1
    threshold = 0.95 * best_value
    gen_95pct = next((i + 1 for i, value in enumerate(history) if value >= threshold), len(history))
    return {"best_generation": best_generation, "gen_95pct": gen_95pct}


def _print_summary(results):
    print("\n3-Way Experiment Summary")
    print("=" * 34)
    for result in results:
        metrics = _convergence_metrics(result["history"])
        print(f"\n{result['name']}")
        print(f"  Best fitness: {result['best']['mission_fitness']:.4f}")
        print(f"  XFOIL calls: {result['xfoil_calls']}")
        print(f"  Runtime: {result['runtime_s']:.2f}s")
        print(f"  Convergence speed (best gen): {metrics['best_generation']}")
        print(f"  Convergence speed (95% gen): {metrics['gen_95pct']}")
        print(f"  ML-guided attempts: {result['guided_attempts']}")
        print(f"  ML-guided acceptances: {result['guided_accepts']}")
        print(f"  ML-guided acceptance rate: {result['guided_acceptance_rate']:.2f}%")


def _save_summary(results):
    with open(SUMMARY_FILE, "w", encoding="utf-8") as handle:
        handle.write("3-WAY EXPERIMENT SUMMARY\n")
        handle.write("========================\n\n")
        for result in results:
            metrics = _convergence_metrics(result["history"])
            handle.write(f"{result['name']}\n")
            handle.write(f"- Best fitness: {result['best']['mission_fitness']:.6f}\n")
            handle.write(f"- XFOIL calls: {result['xfoil_calls']}\n")
            handle.write(f"- Runtime (s): {result['runtime_s']:.4f}\n")
            handle.write(f"- Best generation: {metrics['best_generation']}\n")
            handle.write(f"- 95% convergence generation: {metrics['gen_95pct']}\n")
            handle.write(f"- ML-guided attempts: {result['guided_attempts']}\n")
            handle.write(f"- ML-guided acceptances: {result['guided_accepts']}\n")
            handle.write(
                f"- ML-guided acceptance rate (%): {result['guided_acceptance_rate']:.4f}\n\n"
            )


def _save_plot(results):
    plt.figure(figsize=(10, 6))
    max_len = max(len(result["history"]) for result in results)
    for result in results:
        history = result["history"]
        padded = history + [history[-1]] * (max_len - len(history))
        plt.plot(padded, marker="o", markersize=4, label=result["name"])

    plt.xlabel("Generation")
    plt.ylabel("Best Mission Fitness")
    plt.title("3-Way Convergence Comparison")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()


def run_three_way_experiment(seed=42):
    modes = [
        ("GA only", False, False),
        ("GA + ML (filter)", True, False),
        ("GA + ML (guided)", True, True),
    ]

    results = []
    for name, use_surrogate, guided_search in modes:
        print(f"\nRunning {name}...")
        results.append(
            _run_mode(
                name=name,
                use_surrogate=use_surrogate,
                guided_search=guided_search,
                seed=seed,
            )
        )

    _print_summary(results)
    _save_summary(results)
    _save_plot(results)
    print(f"\nSaved summary to {SUMMARY_FILE}")
    print(f"Saved convergence plot to {PLOT_FILE}")

    return results


if __name__ == "__main__":
    run_three_way_experiment()
