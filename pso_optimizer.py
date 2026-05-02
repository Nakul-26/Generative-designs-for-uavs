import random
import copy
from genetic_optimizer import (
    compute_chord,
    compute_reynolds,
    generate_random_design,
    evaluate_airfoil_details,
    is_valid_reynolds,
    score_design,
    SPAN_MIN, SPAN_MAX, AREA_MIN, AREA_MAX, 
    VELOCITY_MIN, VELOCITY_MAX, BATTERY_MIN_WH, BATTERY_MAX_WH,
    USE_SURROGATE, train_model
)

NUM_PARTICLES = 20
ITERATIONS = 15

W = 0.5
C1 = 1.5
C2 = 1.5

def clamp_design(d):
    d["wing_span"] = max(SPAN_MIN, min(SPAN_MAX, d["wing_span"]))
    d["wing_area"] = max(AREA_MIN, min(AREA_MAX, d["wing_area"]))
    d["velocity"] = max(VELOCITY_MIN, min(VELOCITY_MAX, d["velocity"]))
    d["battery_wh"] = max(BATTERY_MIN_WH, min(BATTERY_MAX_WH, d["battery_wh"]))
    return d

def create_particle():
    design = generate_random_design()
    return {
        "position": design,
        "velocity": {
            "wing_span": random.uniform(-0.1, 0.1),
            "wing_area": random.uniform(-0.01, 0.01),
            "velocity": random.uniform(-1, 1),
            "battery_wh": random.uniform(-10, 10),
        },
        "best_position": copy.deepcopy(design),
        "best_fitness": -1
    }

def update_particle(p, global_best):
    for key in ["wing_span", "wing_area", "velocity", "battery_wh"]:
        r1 = random.random()
        r2 = random.random()
        
        p["velocity"][key] = (
            W * p["velocity"][key]
            + C1 * r1 * (p["best_position"][key] - p["position"][key])
            + C2 * r2 * (global_best[key] - p["position"][key])
        )
        p["position"][key] += p["velocity"][key]

def run_pso():
    print("\n--- Starting PSO Optimization ---")
    model = train_model() if USE_SURROGATE else None
    particles = [create_particle() for _ in range(NUM_PARTICLES)]
    
    global_best = None
    global_best_fitness = -1
    history = []

    for iteration in range(ITERATIONS):
        for p in particles:
            design = clamp_design(p["position"])
            chord = compute_chord(design["wing_area"], design["wing_span"])
            reynolds = compute_reynolds(design["velocity"], chord)
            if not is_valid_reynolds(reynolds):
                fitness = 0
                result = {"mission_fitness": 0}
            else:
                # Airfoil details are needed for scoring.
                airfoil_details = evaluate_airfoil_details(design["airfoil"], model, reynolds=reynolds)
                result = score_design(design, airfoil_details, chord=chord, reynolds=reynolds)
                fitness = result["mission_fitness"]

            if fitness > p["best_fitness"]:
                p["best_fitness"] = fitness
                p["best_position"] = copy.deepcopy(design)

            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best = copy.deepcopy(design)
                global_best["fitness"] = fitness # For compatibility with your suggested print

        for p in particles:
            update_particle(p, global_best)

        history.append(global_best_fitness)
        print(f"PSO Iteration {iteration}: Best Fitness = {global_best_fitness:.2f}")

    return global_best, history

if __name__ == "__main__":
    best, history = run_pso()
    print("\nBest Design Found:")
    print(best)
