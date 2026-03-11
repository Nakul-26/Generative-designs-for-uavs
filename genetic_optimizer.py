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


def evaluate_airfoil(naca, model=None):
    global ml_predictions, ml_skips, xfoil_calls

    if naca in fitness_cache:
        return fitness_cache[naca]

    if model is not None:
        with counter_lock:
            ml_predictions += 1
        mean_ld, uncertainty = predict_ld_with_uncertainty(model, naca)

        if mean_ld + uncertainty < 100:
            print("ML skipped:", naca, "predicted L/D =", mean_ld, "uncertainty =", uncertainty)
            with counter_lock:
                ml_skips += 1
            return mean_ld

    with counter_lock:
        xfoil_calls += 1
    cl, cd = run_xfoil(naca)

    if cl is None or cd is None or cd == 0:
        fitness_cache[naca] = 0
        return 0

    ld = cl / cd

    if cl < 0 or cl > 2.0:
        ld *= 0.2

    if cd < 0.003 or cd > 0.05:
        ld *= 0.2

    if ld > 250:
        ld *= 0.2

    fitness_cache[naca] = ld
    return ld


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
    raw_score_lookup = {
        airfoil: fitness_cache[airfoil]
        for airfoil in population
        if airfoil in fitness_cache
    }

    for airfoil in population:
        if airfoil not in fitness_cache and airfoil not in unique_airfoils:
            unique_airfoils.append(airfoil)

    if unique_airfoils:
        worker_count = min(cpu_count(), len(unique_airfoils))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            scores = list(executor.map(evaluate_airfoil, unique_airfoils, [model] * len(unique_airfoils)))

        for airfoil, score in zip(unique_airfoils, scores):
            raw_score_lookup[airfoil] = score

    scored_population = []

    for airfoil in population:
        raw_score = raw_score_lookup.get(airfoil, 0)
        adjusted_score = raw_score - diversity_penalty(airfoil, population)
        scored_population.append((airfoil, raw_score, adjusted_score))

    return scored_population


def run_ga():
    global xfoil_calls, ml_predictions, ml_skips

    start_time = time.time()
    random.seed(42)
    best_airfoils.clear()
    xfoil_calls = 0
    ml_predictions = 0
    ml_skips = 0
    model = train_model() if USE_SURROGATE else None
    population = [generate_random_naca() for _ in range(POPULATION_SIZE)]
    best_history = []
    stats = []

    for generation in range(GENERATIONS):
        print("\nGeneration:", generation)

        scored_population = evaluate_population(population, model)

        for airfoil, raw_score, adjusted_score in scored_population:
            print(airfoil, "L/D =", raw_score, "fitness =", adjusted_score)

        scored_population.sort(key=lambda item: item[2], reverse=True)

        best = scored_population[0]
        best_history.append(best[1])
        best_airfoils.append((best[0], best[1]))
        stats.append((generation, best[1]))
        print("Best airfoil:", (best[0], best[1]))
        print("Best L/D this generation:", best[1])
        print("Best adjusted fitness this generation:", best[2])

        survivors = [
            airfoil for airfoil, raw_score, adjusted_score in scored_population[: POPULATION_SIZE // 2]
        ]

        new_population = [best[0]] + survivors.copy()

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

    plot_airfoil(best[0])
    print("Saved best airfoil plot to", Path("best_airfoil.png"))

    baseline = "NACA 2412"
    baseline_score = evaluate_airfoil(baseline)
    save_fitness_cache()
    runtime = time.time() - start_time
    print("\nBaseline airfoil:", baseline)
    print("Baseline L/D:", baseline_score)
    print("Best optimized airfoil:", best[0])
    print("Best optimized L/D:", best[1])
    print("\n--- Optimization Statistics ---")
    print("Total XFOIL simulations:", xfoil_calls)
    print("ML predictions:", ml_predictions)
    print("ML skipped designs:", ml_skips)
    print("Cache size:", len(fitness_cache))
    print("Runtime:", runtime, "seconds")

    with open("experiment_summary.txt", "w") as f:
        f.write(f"Use Surrogate: {USE_SURROGATE}\n")
        f.write(f"Baseline L/D: {baseline_score}\n")
        f.write(f"Best Airfoil: {best[0]}\n")
        f.write(f"Best L/D: {best[1]}\n")
        f.write(f"XFOIL calls: {xfoil_calls}\n")
        f.write(f"ML predictions: {ml_predictions}\n")
        f.write(f"ML skips: {ml_skips}\n")
        f.write(f"Cache size: {len(fitness_cache)}\n")
        f.write(f"Runtime: {runtime}\n")
