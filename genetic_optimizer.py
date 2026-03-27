import json
import os
import random
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from threading import Lock

_MPL_CONFIG_DIR = Path(".matplotlib")
_MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR.resolve()))

import matplotlib.pyplot as plt

from airfoil_generator import generate_random_naca
from airfoil_plotter import plot_airfoil
from surrogate_model import predict_ld_with_uncertainty, train_model
from xfoil_runner import run_xfoil


POPULATION_SIZE = 30
GENERATIONS = 15
MUTATION_RATE = 0.6
CACHE_FILE = Path("airfoil_cache.json")
USE_SURROGATE = True
VIS_STATE_FILE = Path("visualization_state.json")
CONTROL_FILE = Path("control.json")
best_airfoils = []
xfoil_calls = 0
ml_skips = 0
ml_predictions = 0
counter_lock = Lock()


def load_fitness_cache():
    if not CACHE_FILE.exists():
        return {}

    with open(CACHE_FILE, "r") as f:
        return json.load(f)


def save_fitness_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(fitness_cache, f, indent=2)


fitness_cache = load_fitness_cache()


def write_visualization_state(state):
    tmp_path = VIS_STATE_FILE.with_suffix(".json.tmp")

    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)

    os.replace(tmp_path, VIS_STATE_FILE)


def is_paused():
    if not CONTROL_FILE.exists():
        return False

    try:
        with open(CONTROL_FILE, "r") as f:
            control = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    return bool(control.get("paused", False))


def wait_if_paused():
    announced_pause = False

    while is_paused():
        if not announced_pause:
            print("Optimizer paused. Waiting for resume...")
            announced_pause = True
        time.sleep(1)

    if announced_pause:
        print("Optimizer resumed.")


def mutate_airfoil(naca):
    """Mutate within valid 4-digit NACA parameter bounds."""
    digits = list(naca.replace("NACA ", ""))
    camber = int(digits[0])
    position = int(digits[1])
    thickness = int("".join(digits[2:]))

    camber += random.choice([-2, -1, 0, 1, 2])
    camber = max(0, min(9, camber))

    position += random.choice([-2, -1, 0, 1, 2])
    position = max(1, min(9, position))

    thickness += random.choice([-3, -2, -1, 0, 1, 2, 3])
    thickness = max(6, min(18, thickness))

    return f"NACA {camber}{position}{thickness:02d}"


def crossover(parent1, parent2):
    d1 = parent1.replace("NACA ", "")
    d2 = parent2.replace("NACA ", "")
    child_digits = ""

    for idx in range(4):
        child_digits += random.choice([d1[idx], d2[idx]])

    return "NACA " + child_digits


def evaluate_airfoil_details(naca, model=None):
    global ml_predictions, ml_skips, xfoil_calls

    if naca in fitness_cache:
        return {
            "score": fitness_cache[naca],
            "evaluation_type": "cached",
            "surrogate_used": False,
            "surrogate_mean_ld": None,
            "surrogate_uncertainty": None,
        }

    surrogate_mean_ld = None
    surrogate_uncertainty = None
    surrogate_used = False
    if model is not None:
        with counter_lock:
            ml_predictions += 1
        surrogate_used = True
        surrogate_mean_ld, surrogate_uncertainty = predict_ld_with_uncertainty(model, naca)

        if surrogate_mean_ld + surrogate_uncertainty < 100:
            print(
                "ML skipped:",
                naca,
                "predicted L/D =",
                surrogate_mean_ld,
                "uncertainty =",
                surrogate_uncertainty,
            )
            with counter_lock:
                ml_skips += 1
            return {
                "score": surrogate_mean_ld,
                "evaluation_type": "skipped",
                "surrogate_used": True,
                "surrogate_mean_ld": surrogate_mean_ld,
                "surrogate_uncertainty": surrogate_uncertainty,
            }

    with counter_lock:
        xfoil_calls += 1
    cl, cd = run_xfoil(naca)

    if cl is None or cd is None or cd == 0:
        fitness_cache[naca] = 0
        return {
            "score": 0,
            "evaluation_type": "simulated",
            "surrogate_used": surrogate_used,
            "surrogate_mean_ld": surrogate_mean_ld,
            "surrogate_uncertainty": surrogate_uncertainty,
        }

    ld = cl / cd

    if cl < 0 or cl > 2.0:
        ld *= 0.2

    if cd < 0.003 or cd > 0.05:
        ld *= 0.2

    if ld > 250:
        ld *= 0.2

    fitness_cache[naca] = ld
    return {
        "score": ld,
        "evaluation_type": "simulated",
        "surrogate_used": surrogate_used,
        "surrogate_mean_ld": surrogate_mean_ld,
        "surrogate_uncertainty": surrogate_uncertainty,
    }


def evaluate_airfoil(naca, model=None):
    return evaluate_airfoil_details(naca, model)["score"]


def diversity_penalty(naca, population):
    digits = naca.replace("NACA ", "")
    penalty = 0

    for other in population:
        other_digits = other.replace("NACA ", "")
        diff = sum(a != b for a, b in zip(digits, other_digits))

        if diff <= 1:
            penalty += 10

    return penalty


def evaluate_population(population, model=None):
    unique_airfoils = []
    raw_score_lookup = {}
    details_lookup = {}

    for airfoil in population:
        if airfoil in fitness_cache:
            raw_score_lookup[airfoil] = fitness_cache[airfoil]
            details_lookup[airfoil] = {
                "evaluation_type": "cached",
                "surrogate_used": False,
                "surrogate_mean_ld": None,
                "surrogate_uncertainty": None,
            }

    for airfoil in population:
        if airfoil not in fitness_cache and airfoil not in unique_airfoils:
            unique_airfoils.append(airfoil)

    if unique_airfoils:
        worker_count = min(cpu_count(), len(unique_airfoils))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(
                executor.map(
                    evaluate_airfoil_details,
                    unique_airfoils,
                    [model] * len(unique_airfoils),
                )
            )

        for airfoil, result in zip(unique_airfoils, results):
            raw_score_lookup[airfoil] = result["score"]
            details_lookup[airfoil] = {
                "evaluation_type": result["evaluation_type"],
                "surrogate_used": result["surrogate_used"],
                "surrogate_mean_ld": result["surrogate_mean_ld"],
                "surrogate_uncertainty": result["surrogate_uncertainty"],
            }

    scored_population = []

    for airfoil in population:
        raw_score = raw_score_lookup.get(airfoil, 0)
        adjusted_score = raw_score - diversity_penalty(airfoil, population)
        details = details_lookup.get(
            airfoil,
            {
                "evaluation_type": "unknown",
                "surrogate_used": False,
                "surrogate_mean_ld": None,
                "surrogate_uncertainty": None,
            },
        )
        scored_population.append(
            {
                "airfoil": airfoil,
                "raw_score": raw_score,
                "adjusted_score": adjusted_score,
                **details,
            }
        )

    return scored_population


def run_ga():
    global xfoil_calls, ml_predictions, ml_skips

    start_time = time.time()
    random.seed(42)
    best_airfoils.clear()
    xfoil_calls = 0
    ml_predictions = 0
    ml_skips = 0
    if not CONTROL_FILE.exists():
        with open(CONTROL_FILE, "w") as f:
            json.dump({"paused": False}, f, indent=2)
    model = train_model() if USE_SURROGATE else None
    population = [generate_random_naca() for _ in range(POPULATION_SIZE)]
    best_history = []
    stats = []

    write_visualization_state(
        {
            "status": "running",
            "generation": -1,
            "generations_total": GENERATIONS,
            "best_airfoil": None,
            "best_ld": None,
            "best_history": [],
            "population": [],
            "xfoil_calls": xfoil_calls,
            "ml_predictions": ml_predictions,
            "ml_skips": ml_skips,
        }
    )

    for generation in range(GENERATIONS):
        wait_if_paused()
        print("\nGeneration:", generation)
        generation_xfoil_before = xfoil_calls
        generation_predictions_before = ml_predictions
        generation_skips_before = ml_skips

        scored_population = evaluate_population(population, model)

        for entry in scored_population:
            print(
                entry["airfoil"],
                "L/D =",
                entry["raw_score"],
                "fitness =",
                entry["adjusted_score"],
                "type =",
                entry["evaluation_type"],
            )

        scored_population.sort(key=lambda item: item["adjusted_score"], reverse=True)

        best = scored_population[0]
        best_history.append(best["raw_score"])
        best_airfoils.append((best["airfoil"], best["raw_score"]))
        stats.append((generation, best["raw_score"]))
        print("Best airfoil:", (best["airfoil"], best["raw_score"]))
        print("Best L/D this generation:", best["raw_score"])
        print("Best adjusted fitness this generation:", best["adjusted_score"])

        source_counts = {"simulated": 0, "cached": 0, "skipped": 0, "unknown": 0}
        for entry in scored_population:
            source_counts[entry["evaluation_type"]] = source_counts.get(entry["evaluation_type"], 0) + 1

        write_visualization_state(
            {
                "status": "running",
                "generation": generation,
                "generations_total": GENERATIONS,
                "best_airfoil": best["airfoil"],
                "best_ld": best["raw_score"],
                "best_adjusted_fitness": best["adjusted_score"],
                "best_history": best_history,
                "population": [
                    {
                        "airfoil": entry["airfoil"],
                        "ld": entry["raw_score"],
                        "adjusted_fitness": entry["adjusted_score"],
                        "evaluation_type": entry["evaluation_type"],
                        "surrogate_used": entry["surrogate_used"],
                        "surrogate_mean_ld": entry["surrogate_mean_ld"],
                        "surrogate_uncertainty": entry["surrogate_uncertainty"],
                    }
                    for entry in scored_population
                ],
                "source_counts": source_counts,
                "xfoil_calls": xfoil_calls,
                "ml_predictions": ml_predictions,
                "ml_skips": ml_skips,
                "generation_xfoil_calls": xfoil_calls - generation_xfoil_before,
                "generation_predictions": ml_predictions - generation_predictions_before,
                "generation_ml_skips": ml_skips - generation_skips_before,
            }
        )

        survivors = [
            entry["airfoil"] for entry in scored_population[: POPULATION_SIZE // 2]
        ]

        new_population = [best["airfoil"]] + survivors.copy()

        while len(new_population) < POPULATION_SIZE:
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            child = crossover(parent1, parent2)

            if random.random() < MUTATION_RATE:
                child = mutate_airfoil(child)

            new_population.append(child)

        population = new_population
        save_fitness_cache()

        if USE_SURROGATE and (generation + 1) % 2 == 0:
            model = train_model()

    plt.figure()
    plt.plot(best_history, marker="o")
    plt.title("GA Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best L/D")
    plt.grid(True)
    plot_path = Path("ga_convergence.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved convergence plot to", plot_path)

    with open("best_airfoils.txt", "w") as f:
        for airfoil, score in best_airfoils:
            f.write(f"{airfoil}  L/D={score}\n")

    with open("optimization_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_LD"])
        writer.writerows(stats)

    plot_airfoil(best["airfoil"])
    print("Saved best airfoil plot to", Path("best_airfoil.png"))

    baseline = "NACA 2412"
    baseline_score = evaluate_airfoil(baseline)
    save_fitness_cache()
    runtime = time.time() - start_time
    print("\nBaseline airfoil:", baseline)
    print("Baseline L/D:", baseline_score)
    print("Best optimized airfoil:", best["airfoil"])
    print("Best optimized L/D:", best["raw_score"])
    print("\n--- Optimization Statistics ---")
    print("Total XFOIL simulations:", xfoil_calls)
    print("ML predictions:", ml_predictions)
    print("ML skipped designs:", ml_skips)
    print("Cache size:", len(fitness_cache))
    print("Runtime:", runtime, "seconds")

    write_visualization_state(
        {
            "status": "completed",
            "generation": GENERATIONS - 1,
            "generations_total": GENERATIONS,
            "best_airfoil": best["airfoil"],
            "best_ld": best["raw_score"],
            "best_adjusted_fitness": best["adjusted_score"],
            "best_history": best_history,
            "population": [
                {
                    "airfoil": entry["airfoil"],
                    "ld": entry["raw_score"],
                    "adjusted_fitness": entry["adjusted_score"],
                    "evaluation_type": entry["evaluation_type"],
                    "surrogate_used": entry["surrogate_used"],
                    "surrogate_mean_ld": entry["surrogate_mean_ld"],
                    "surrogate_uncertainty": entry["surrogate_uncertainty"],
                }
                for entry in scored_population
            ],
            "source_counts": source_counts,
            "xfoil_calls": xfoil_calls,
            "ml_predictions": ml_predictions,
            "ml_skips": ml_skips,
            "runtime_seconds": runtime,
        }
    )

    with open("experiment_summary.txt", "w") as f:
        f.write(f"Use Surrogate: {USE_SURROGATE}\n")
        f.write(f"Baseline L/D: {baseline_score}\n")
        f.write(f"Best Airfoil: {best['airfoil']}\n")
        f.write(f"Best L/D: {best['raw_score']}\n")
        f.write(f"XFOIL calls: {xfoil_calls}\n")
        f.write(f"ML predictions: {ml_predictions}\n")
        f.write(f"ML skips: {ml_skips}\n")
        f.write(f"Cache size: {len(fitness_cache)}\n")
        f.write(f"Runtime: {runtime}\n")
