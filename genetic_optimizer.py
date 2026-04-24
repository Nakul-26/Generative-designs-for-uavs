import csv
import json
import math
import os
import random
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

MISSION = {
    "payload_weight": 5.0,
    "target_flight_time": 30.0,
    "target_speed": 15.0,
}

SPAN_MIN = 0.5
SPAN_MAX = 3.0
AREA_MIN = 0.1
AREA_MAX = 1.0
VELOCITY_MIN = 5.0
VELOCITY_MAX = 25.0
RHO = 1.225
STRUCTURE_WEIGHT = 4.0
BATTERY_WEIGHT_PER_WH = 0.02
BATTERY_MIN_WH = 100.0
BATTERY_MAX_WH = 500.0
DEFAULT_BATTERY_WH = 200.0
WEIGHT = MISSION["payload_weight"] + STRUCTURE_WEIGHT + BATTERY_WEIGHT_PER_WH * DEFAULT_BATTERY_WH
OSWALD_EFFICIENCY = 0.85

best_designs = []
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
    with open(VIS_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_mission():
    if not CONTROL_FILE.exists():
        return dict(MISSION)

    try:
        with open(CONTROL_FILE, "r") as f:
            control = json.load(f)
    except (OSError, json.JSONDecodeError):
        return dict(MISSION)

    mission = control.get("mission", MISSION)
    return {
        "payload_weight": float(mission.get("payload_weight", MISSION["payload_weight"])),
        "target_flight_time": float(
            mission.get("target_flight_time", MISSION["target_flight_time"])
        ),
        "target_speed": float(mission.get("target_speed", MISSION["target_speed"])),
    }


def apply_mission(mission):
    global WEIGHT
    MISSION.update(
        {
            "payload_weight": float(mission.get("payload_weight", MISSION["payload_weight"])),
            "target_flight_time": float(
                mission.get("target_flight_time", MISSION["target_flight_time"])
            ),
            "target_speed": float(mission.get("target_speed", MISSION["target_speed"])),
        }
    )
    WEIGHT = MISSION["payload_weight"] + STRUCTURE_WEIGHT + BATTERY_WEIGHT_PER_WH * DEFAULT_BATTERY_WH


def is_running():
    if not CONTROL_FILE.exists():
        return True

    try:
        with open(CONTROL_FILE, "r") as f:
            control = json.load(f)
    except (OSError, json.JSONDecodeError):
        return True

    return bool(control.get("running", True))


def wait_until_running():
    announced_stop = False

    while not is_running():
        if not announced_stop:
            print("Optimizer stopped. Waiting to start...")
            announced_stop = True
        time.sleep(1)

    if announced_stop:
        print("Optimizer restarted.")


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def round_design_value(value):
    return round(value, 3)


def round_velocity(value):
    return round(value, 2)


def clamp_velocity(value):
    return clamp(value, VELOCITY_MIN, VELOCITY_MAX)


def round_battery_wh(value):
    return round(value, 1)


def clamp_battery_wh(value):
    return clamp(value, BATTERY_MIN_WH, BATTERY_MAX_WH)


def battery_weight(battery_wh):
    return BATTERY_WEIGHT_PER_WH * battery_wh


def design_weight(battery_wh):
    return MISSION["payload_weight"] + STRUCTURE_WEIGHT + battery_weight(battery_wh)


def battery_energy_j(battery_wh):
    return battery_wh * 3600.0


def create_design(airfoil, wing_span, wing_area, velocity, battery_wh):
    return {
        "airfoil": airfoil,
        "wing_span": round_design_value(clamp(wing_span, SPAN_MIN, SPAN_MAX)),
        "wing_area": round_design_value(clamp(wing_area, AREA_MIN, AREA_MAX)),
        "velocity": round_velocity(clamp_velocity(velocity)),
        "battery_wh": round_battery_wh(clamp_battery_wh(battery_wh)),
    }


def clone_design(design):
    return create_design(
        design["airfoil"],
        design["wing_span"],
        design["wing_area"],
        design["velocity"],
        design.get("battery_wh", DEFAULT_BATTERY_WH),
    )


def format_design_label(design):
    return (
        f"{design['airfoil']} | "
        f"b={design['wing_span']:.2f}m | "
        f"S={design['wing_area']:.2f}m^2"
        f" | v={design['velocity']:.2f}m/s"
        f" | batt={design['battery_wh']:.1f}Wh"
    )


def generate_random_design():
    target_speed = MISSION["target_speed"]
    return create_design(
        generate_random_naca(),
        random.uniform(SPAN_MIN, SPAN_MAX),
        random.uniform(AREA_MIN, AREA_MAX),
        random.uniform(target_speed * 0.8, target_speed * 1.2),
        random.uniform(BATTERY_MIN_WH, BATTERY_MAX_WH),
    )


def mutate_airfoil(naca):
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


def mutate_design(design):
    mutated = clone_design(design)

    if random.random() < 0.65:
        mutated["airfoil"] = mutate_airfoil(mutated["airfoil"])

    mutated["wing_span"] = round_design_value(
        clamp(mutated["wing_span"] + random.uniform(-0.25, 0.25), SPAN_MIN, SPAN_MAX)
    )
    mutated["wing_area"] = round_design_value(
        clamp(mutated["wing_area"] + random.uniform(-0.08, 0.08), AREA_MIN, AREA_MAX)
    )
    mutated["velocity"] = round_velocity(
        clamp_velocity(mutated["velocity"] + random.uniform(-1.5, 1.5))
    )
    mutated["battery_wh"] = round_battery_wh(
        clamp_battery_wh(mutated["battery_wh"] + random.uniform(-40.0, 40.0))
    )

    return mutated


def crossover_airfoils(parent1, parent2):
    d1 = parent1.replace("NACA ", "")
    d2 = parent2.replace("NACA ", "")
    child_digits = ""

    for idx in range(4):
        child_digits += random.choice([d1[idx], d2[idx]])

    return "NACA " + child_digits


def crossover_design(parent1, parent2):
    return create_design(
        crossover_airfoils(parent1["airfoil"], parent2["airfoil"]),
        random.choice(
            [
                parent1["wing_span"],
                parent2["wing_span"],
                (parent1["wing_span"] + parent2["wing_span"]) / 2,
            ]
        ),
        random.choice(
            [
                parent1["wing_area"],
                parent2["wing_area"],
                (parent1["wing_area"] + parent2["wing_area"]) / 2,
            ]
        ),
        random.choice(
            [
                parent1["velocity"],
                parent2["velocity"],
                (parent1["velocity"] + parent2["velocity"]) / 2,
            ]
        ),
        random.choice(
            [
                parent1["battery_wh"],
                parent2["battery_wh"],
                (parent1["battery_wh"] + parent2["battery_wh"]) / 2,
            ]
        ),
    )


def get_cached_airfoil_entry(naca):
    entry = fitness_cache.get(naca)
    if isinstance(entry, dict):
        return {
            "cl": entry.get("cl"),
            "cd": entry.get("cd"),
            "ld": entry.get("ld", 0),
        }

    return None


def evaluate_airfoil_details(naca, model=None):
    global ml_predictions, xfoil_calls

    cached_entry = get_cached_airfoil_entry(naca)
    if cached_entry is not None and cached_entry["cl"] is not None and cached_entry["cd"] is not None:
        return {
            "score": cached_entry["ld"],
            "cl": cached_entry["cl"],
            "cd": cached_entry["cd"],
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

    with counter_lock:
        xfoil_calls += 1
    cl, cd = run_xfoil(naca)

    if cl is None or cd is None or cd == 0:
        fitness_cache[naca] = {"cl": 0, "cd": None, "ld": 0}
        return {
            "score": 0,
            "cl": 0,
            "cd": None,
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

    fitness_cache[naca] = {"cl": cl, "cd": cd, "ld": ld}
    return {
        "score": ld,
        "cl": cl,
        "cd": cd,
        "evaluation_type": "simulated",
        "surrogate_used": surrogate_used,
        "surrogate_mean_ld": surrogate_mean_ld,
        "surrogate_uncertainty": surrogate_uncertainty,
    }


def evaluate_airfoil(naca, model=None):
    return evaluate_airfoil_details(naca, model)["score"]


def induced_drag_coefficient(cl, aspect_ratio):
    if aspect_ratio <= 0:
        return 1.0

    return (cl ** 2) / (math.pi * OSWALD_EFFICIENCY * aspect_ratio)


def score_design(design, airfoil_details):
    wing_area = design["wing_area"]
    wing_span = design["wing_span"]
    battery_wh = design["battery_wh"]
    weight = design_weight(battery_wh)
    aspect_ratio = wing_span ** 2 / wing_area
    cl = airfoil_details.get("cl") or 0
    base_cd = airfoil_details.get("cd")
    base_ld = airfoil_details.get("score", 0)
    target_time = MISSION["target_flight_time"]

    if base_cd in (None, 0) or cl <= 0:
        return {
            "score": 0,
            "mission_fitness": 0,
            "time_score": 0,
            "lift": 0,
            "drag": 0,
            "power": None,
            "weight": weight,
            "battery_weight": battery_weight(battery_wh),
            "battery_energy_j": battery_energy_j(battery_wh),
            "flight_time_s": None,
            "flight_time_min": None,
            "lift_margin": -weight,
            "constraint_satisfied": False,
            "aspect_ratio": aspect_ratio,
            "base_ld": base_ld,
            "base_cd": base_cd,
            "total_cd": None,
            "realism_penalty": 1.0,
        }

    total_cd = base_cd + induced_drag_coefficient(cl, aspect_ratio)
    velocity = design["velocity"]
    q = 0.5 * RHO * velocity ** 2
    lift = q * cl * wing_area
    drag = q * total_cd * wing_area
    power = drag * velocity
    ld = lift / drag if drag > 0 else 0
    constraint_satisfied = lift >= weight
    flight_time_s = battery_energy_j(battery_wh) / power if power > 0 else None
    flight_time_min = flight_time_s / 60.0 if flight_time_s is not None else None
    time_score = 0
    if flight_time_min is not None and target_time > 0:
        ratio = flight_time_min / target_time
        if ratio < 1:
            time_score = ratio
        elif ratio <= 1.5:
            time_score = 1.0
        else:
            time_score = max(0.5, 1.5 - (ratio - 1.5))

    mission_fitness = ld + 10.0 * time_score
    if not constraint_satisfied:
        mission_fitness *= 0.1

    realism_penalty = 1.0
    if ld > 150:
        realism_penalty *= 0.5
    if velocity < 5 or velocity > 25:
        realism_penalty *= 0.5
    if wing_area < 0.05 or wing_area > 1.0:
        realism_penalty *= 0.5
    if aspect_ratio < 4 or aspect_ratio > 15:
        realism_penalty *= 0.5
    if power < 5:
        realism_penalty *= 0.5

    mission_fitness *= realism_penalty

    return {
        "score": ld,
        "mission_fitness": mission_fitness,
        "time_score": time_score,
        "lift": lift,
        "drag": drag,
        "power": power,
        "weight": weight,
        "battery_weight": battery_weight(battery_wh),
        "battery_energy_j": battery_energy_j(battery_wh),
        "flight_time_s": flight_time_s,
        "flight_time_min": flight_time_min,
        "lift_margin": lift - weight,
        "constraint_satisfied": constraint_satisfied,
        "aspect_ratio": aspect_ratio,
        "base_ld": base_ld,
        "base_cd": base_cd,
        "total_cd": total_cd,
        "dynamic_pressure": q,
        "realism_penalty": realism_penalty,
    }


def diversity_penalty(design, population):
    digits = design["airfoil"].replace("NACA ", "")
    penalty = 0

    for other in population:
        if other == design:
            continue

        other_digits = other["airfoil"].replace("NACA ", "")
        diff = sum(a != b for a, b in zip(digits, other_digits))
        span_gap = abs(design["wing_span"] - other["wing_span"])
        area_gap = abs(design["wing_area"] - other["wing_area"])
        battery_gap = abs(design["battery_wh"] - other["battery_wh"])

        if diff <= 1 and span_gap < 0.2 and area_gap < 0.08 and battery_gap < 25.0:
            penalty += 10

    return penalty


def dominates(a, b):
    a_power = a.get("power")
    b_power = b.get("power")
    a_time = a.get("flight_time_min")
    b_time = b.get("flight_time_min")

    if a_power is None:
        return False
    if b_power is None:
        return True

    a_time = a_time if a_time is not None else 0
    b_time = b_time if b_time is not None else 0

    return (
        a.get("ld", 0) >= b.get("ld", 0)
        and a_time >= b_time
        and a_power <= b_power
        and (
            a.get("ld", 0) > b.get("ld", 0)
            or a_time > b_time
            or a_power < b_power
        )
    )


def build_pareto_front(scored_population):
    pareto_front = []
    seen_signatures = set()

    for candidate in scored_population:
        dominated = False
        for other in scored_population:
            if other is candidate:
                continue
            if dominates(other, candidate):
                dominated = True
                break

        if not dominated:
            signature = (
                candidate["airfoil"],
                candidate["wing_span"],
                candidate["wing_area"],
                candidate["velocity"],
                candidate["battery_wh"],
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            pareto_front.append(
                {
                    "airfoil": candidate["airfoil"],
                    "label": candidate["label"],
                    "ld": candidate["raw_score"],
                    "flight_time_min": candidate["flight_time_min"],
                    "power": candidate["power"],
                    "wing_span": candidate["wing_span"],
                    "wing_area": candidate["wing_area"],
                    "velocity": candidate["velocity"],
                    "battery_wh": candidate["battery_wh"],
                    "lift": candidate["lift"],
                    "weight": candidate["weight"],
                    "mission_fitness": candidate["mission_fitness"],
                    "adjusted_fitness": candidate["adjusted_score"],
                }
            )

    pareto_front.sort(
        key=lambda item: (
            -(item["ld"] or 0),
            -(item["flight_time_min"] or 0),
            item["power"] if item["power"] is not None else float("inf"),
        )
    )
    return pareto_front


def evaluate_population(population, model=None):
    unique_airfoils = []
    airfoil_details_lookup = {}

    for design in population:
        airfoil = design["airfoil"]
        cached_entry = get_cached_airfoil_entry(airfoil)
        if cached_entry is not None and cached_entry["cl"] is not None and cached_entry["cd"] is not None:
            airfoil_details_lookup[airfoil] = {
                "score": cached_entry["ld"],
                "cl": cached_entry["cl"],
                "cd": cached_entry["cd"],
                "evaluation_type": "cached",
                "surrogate_used": False,
                "surrogate_mean_ld": None,
                "surrogate_uncertainty": None,
            }

    for design in population:
        airfoil = design["airfoil"]
        if airfoil not in airfoil_details_lookup and airfoil not in unique_airfoils:
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
            airfoil_details_lookup[airfoil] = result

    scored_population = []

    for design in population:
        airfoil_details = airfoil_details_lookup.get(
            design["airfoil"],
            {
                "score": 0,
                "cl": 0,
                "cd": None,
                "evaluation_type": "unknown",
                "surrogate_used": False,
                "surrogate_mean_ld": None,
                "surrogate_uncertainty": None,
            },
        )
        design_score = score_design(design, airfoil_details)
        adjusted_score = design_score["mission_fitness"] - diversity_penalty(design, population)
        scored_population.append(
            {
                "airfoil": design["airfoil"],
                "wing_span": design["wing_span"],
                "wing_area": design["wing_area"],
                "label": format_design_label(design),
                "velocity": design["velocity"],
                "battery_wh": design["battery_wh"],
                "raw_score": design_score["score"],
                "mission_fitness": design_score["mission_fitness"],
                "time_score": design_score["time_score"],
                "adjusted_score": adjusted_score,
                "lift": design_score["lift"],
                "drag": design_score["drag"],
                "power": design_score["power"],
                "weight": design_score["weight"],
                "battery_weight": design_score["battery_weight"],
                "battery_energy_j": design_score["battery_energy_j"],
                "flight_time_s": design_score["flight_time_s"],
                "flight_time_min": design_score["flight_time_min"],
                "lift_margin": design_score["lift_margin"],
                "constraint_satisfied": design_score["constraint_satisfied"],
                "aspect_ratio": design_score["aspect_ratio"],
                "base_ld": design_score["base_ld"],
                "base_cd": design_score["base_cd"],
                "total_cd": design_score["total_cd"],
                "dynamic_pressure": design_score.get("dynamic_pressure"),
                "evaluation_type": airfoil_details["evaluation_type"],
                "surrogate_used": airfoil_details["surrogate_used"],
                "surrogate_mean_ld": airfoil_details["surrogate_mean_ld"],
                "surrogate_uncertainty": airfoil_details["surrogate_uncertainty"],
            }
        )

    return scored_population


def population_state(scored_population):
    return [
        {
            "airfoil": entry["airfoil"],
            "label": entry["label"],
            "wing_span": entry["wing_span"],
            "wing_area": entry["wing_area"],
            "velocity": entry["velocity"],
            "battery_wh": entry["battery_wh"],
            "ld": entry["raw_score"],
            "mission_fitness": entry["mission_fitness"],
            "time_score": entry["time_score"],
            "adjusted_fitness": entry["adjusted_score"],
            "lift": entry["lift"],
            "drag": entry["drag"],
            "power": entry["power"],
            "weight": entry["weight"],
            "battery_weight": entry["battery_weight"],
            "battery_energy_j": entry["battery_energy_j"],
            "flight_time_s": entry["flight_time_s"],
            "flight_time_min": entry["flight_time_min"],
            "lift_margin": entry["lift_margin"],
            "constraint_satisfied": entry["constraint_satisfied"],
            "aspect_ratio": entry["aspect_ratio"],
            "base_ld": entry["base_ld"],
            "base_cd": entry["base_cd"],
            "total_cd": entry["total_cd"],
            "evaluation_type": entry["evaluation_type"],
            "surrogate_used": entry["surrogate_used"],
            "surrogate_mean_ld": entry["surrogate_mean_ld"],
            "surrogate_uncertainty": entry["surrogate_uncertainty"],
        }
        for entry in scored_population
    ]


def pareto_state(pareto_front):
    return [
        {
            "airfoil": entry["airfoil"],
            "label": entry["label"],
            "ld": entry["ld"],
            "flight_time_min": entry["flight_time_min"],
            "power": entry["power"],
            "wing_span": entry["wing_span"],
            "wing_area": entry["wing_area"],
            "velocity": entry["velocity"],
            "battery_wh": entry["battery_wh"],
            "lift": entry["lift"],
            "weight": entry["weight"],
            "mission_fitness": entry["mission_fitness"],
            "adjusted_fitness": entry["adjusted_fitness"],
        }
        for entry in pareto_front
    ]


def run_ga():
    global xfoil_calls, ml_predictions, ml_skips

    start_time = time.time()
    random.seed(42)
    best_designs.clear()
    xfoil_calls = 0
    ml_predictions = 0
    ml_skips = 0
    apply_mission(load_mission())
    if not CONTROL_FILE.exists():
        with open(CONTROL_FILE, "w") as f:
            json.dump({"running": True, "mission": dict(MISSION)}, f, indent=2)
    model = train_model() if USE_SURROGATE else None
    population = [generate_random_design() for _ in range(POPULATION_SIZE)]
    best_history = []
    stats = []

    write_visualization_state(
        {
            "status": "running",
            "generation": 0,
            "generations_total": GENERATIONS,
            "best_airfoil": None,
            "best_span": None,
            "best_area": None,
            "best_velocity": None,
            "best_battery_wh": None,
            "best_dynamic_pressure": 0,
            "best_feasible": None,
            "best_lift": None,
            "best_weight": None,
            "best_lift_margin": None,
            "best_drag": None,
            "best_power": None,
            "best_ld": None,
            "best_mission_fitness": None,
            "best_time_score": None,
            "best_adjusted_fitness": None,
            "pareto_front": [],
            "pareto_front_count": 0,
            "mission_payload": MISSION["payload_weight"],
            "mission_time": MISSION["target_flight_time"],
            "mission_speed": MISSION["target_speed"],
            "weight_target": WEIGHT,
            "dynamic_pressure": 0,
            "best_history": [],
            "population": [],
            "source_counts": {"simulated": 0, "cached": 0, "skipped": 0, "unknown": 0},
            "xfoil_calls": xfoil_calls,
            "ml_predictions": ml_predictions,
            "ml_skips": ml_skips,
            "generation_xfoil_calls": 0,
            "generation_predictions": 0,
            "generation_ml_skips": 0,
            "battery_capacity_Wh": None,
            "best_battery_weight": None,
            "best_battery_energy_J": None,
            "best_flight_time_s": None,
            "best_flight_time_min": None,
        }
    )

    for generation in range(GENERATIONS):
        apply_mission(load_mission())
        wait_until_running()
        print("\nGeneration:", generation)
        generation_xfoil_before = xfoil_calls
        generation_predictions_before = ml_predictions
        generation_skips_before = ml_skips

        scored_population = evaluate_population(population, model)
        pareto_front = build_pareto_front(scored_population)

        for entry in scored_population:
            print(
                entry["label"],
                "L/D =",
                round(entry["raw_score"], 3),
                "lift =",
                round(entry["lift"], 3),
                "fitness =",
                round(entry["adjusted_score"], 3),
                "type =",
                entry["evaluation_type"],
            )

        scored_population.sort(key=lambda item: item["adjusted_score"], reverse=True)

        best = scored_population[0]
        best_history.append(best["mission_fitness"])
        best_designs.append((best["label"], best["raw_score"]))
        stats.append(
            (
                generation,
                best["raw_score"],
                best["mission_fitness"],
                best["adjusted_score"],
                best["wing_span"],
                best["wing_area"],
                best["velocity"],
                best["battery_wh"],
                best["lift"],
            )
        )
        print("Best wing design:", best["label"])
        print("Best L/D this generation:", best["raw_score"])
        print("Best mission fitness this generation:", best["mission_fitness"])
        print("Best adjusted fitness this generation:", best["adjusted_score"])
        print("Pareto front size:", len(pareto_front))
        print("Lift / target:", best["lift"], "/", best["weight"])
        print("Speed (m/s):", best["velocity"])
        print("Battery (Wh):", best["battery_wh"])

        source_counts = {"simulated": 0, "cached": 0, "skipped": 0, "unknown": 0}
        for entry in scored_population:
            source_counts[entry["evaluation_type"]] = source_counts.get(entry["evaluation_type"], 0) + 1

        best_weight = best["weight"]
        feasible = best["lift"] >= best_weight
        lift_margin = (best["lift"] - best_weight) / best_weight * 100
        best_power_val = best.get("power")
        if best_power_val is None and best.get("drag") is not None and best.get("velocity") is not None:
            best_power_val = best.get("drag") * best.get("velocity")
        best_flight_time_s = best.get("flight_time_s")
        best_flight_time_min = best.get("flight_time_min")
        if best_flight_time_min is not None:
            best_flight_time_min = min(best_flight_time_min, 300.0)
        write_visualization_state(
            {
                "status": "running",
                "generation": generation,
                "generations_total": GENERATIONS,
                "best_airfoil": best["airfoil"],
                "best_span": best["wing_span"],
                "best_area": best["wing_area"],
                "best_velocity": best["velocity"],
                "best_battery_wh": best["battery_wh"],
                "best_dynamic_pressure": best.get("dynamic_pressure"),
                "best_feasible": feasible,
                "best_lift": best["lift"],
                "best_weight": best_weight,
                "best_lift_margin": lift_margin,
                "best_drag": best["drag"],
                "best_power": best_power_val if best_power_val is not None else (best.get("drag") * best.get("velocity") if best.get("drag") is not None and best.get("velocity") is not None else None),
                "battery_capacity_Wh": best["battery_wh"],
                "best_battery_weight": best["battery_weight"],
                "best_battery_energy_J": best["battery_energy_j"],
                "best_flight_time_s": best_flight_time_s,
                "best_flight_time_min": best_flight_time_min,
                "best_ld": best["raw_score"],
                "best_mission_fitness": best["mission_fitness"],
                "best_time_score": best["time_score"],
                "best_adjusted_fitness": best["adjusted_score"],
                "pareto_front": pareto_state(pareto_front),
                "pareto_front_count": len(pareto_front),
                "mission_payload": MISSION["payload_weight"],
                "mission_time": MISSION["target_flight_time"],
                "mission_speed": MISSION["target_speed"],
                "weight_target": WEIGHT,
                "dynamic_pressure": best.get("dynamic_pressure"),
                "best_history": best_history,
                "population": population_state(scored_population),
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
            create_design(
                entry["airfoil"],
                entry["wing_span"],
                entry["wing_area"],
                entry["velocity"],
                entry["battery_wh"],
            )
            for entry in scored_population[: POPULATION_SIZE // 2]
        ]

        new_population = [
            create_design(
                best["airfoil"],
                best["wing_span"],
                best["wing_area"],
                best["velocity"],
                best["battery_wh"],
            )
        ] + survivors.copy()

        while len(new_population) < POPULATION_SIZE:
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            child = crossover_design(parent1, parent2)

            if random.random() < MUTATION_RATE:
                child = mutate_design(child)

            new_population.append(child)

        population = new_population
        save_fitness_cache()

        if USE_SURROGATE and (generation + 1) % 2 == 0:
            model = train_model()

    plt.figure()
    plt.plot(best_history, marker="o")
    plt.title("GA Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best Mission Fitness")
    plt.grid(True)
    plot_path = Path("ga_convergence.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved convergence plot to", plot_path)

    with open("best_airfoils.txt", "w") as f:
        for design_label, score in best_designs:
            f.write(f"{design_label}  L/D={score}\n")

    with open("optimization_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "generation",
                "best_LD",
                "best_mission_fitness",
                "best_adjusted_fitness",
                "wing_span_m",
                "wing_area_m2",
                "velocity_m_s",
                "battery_Wh",
                "lift",
            ]
        )
        writer.writerows(stats)

    plot_airfoil(best["airfoil"])
    print("Saved best airfoil plot to", Path("best_airfoil.png"))

    pareto_front = build_pareto_front(scored_population)
    baseline_design = create_design(
        "NACA 2412",
        1.5,
        0.4,
        MISSION["target_speed"],
        DEFAULT_BATTERY_WH,
    )
    baseline_airfoil_details = evaluate_airfoil_details(baseline_design["airfoil"], model)
    baseline_result = score_design(baseline_design, baseline_airfoil_details)
    save_fitness_cache()
    runtime = time.time() - start_time
    print("\nBaseline wing:", format_design_label(baseline_design))
    print("Baseline L/D:", baseline_result["score"])
    print("Baseline lift:", baseline_result["lift"])
    print("Best optimized wing:", best["label"])
    print("Best optimized L/D:", best["raw_score"])
    print("Best optimized mission fitness:", best["mission_fitness"])
    print("Best optimized adjusted fitness:", best["adjusted_score"])
    print("Pareto front size:", len(pareto_front))
    print("Best optimized lift:", best["lift"])
    print("Best optimized speed (m/s):", best["velocity"])
    print("Best optimized battery (Wh):", best["battery_wh"])
    print("\n--- Optimization Statistics ---")
    print("Total XFOIL simulations:", xfoil_calls)
    print("ML predictions:", ml_predictions)
    print("ML skipped designs:", ml_skips)
    print("Cache size:", len(fitness_cache))
    print("Runtime:", runtime, "seconds")

    best_dynamic_pressure = best.get("dynamic_pressure", 0)
    best_weight = best["weight"]
    feasible = best["lift"] >= best_weight
    lift_margin = (best["lift"] - best_weight) / best_weight * 100
    best_power_val = best.get("power")
    if best_power_val is None and best.get("drag") is not None and best.get("velocity") is not None:
        best_power_val = best.get("drag") * best.get("velocity")
    best_flight_time_s = best.get("flight_time_s")
    best_flight_time_min = best.get("flight_time_min")
    if best_flight_time_s is not None:
        best_flight_time_min = min(best_flight_time_min, 300.0)

    write_visualization_state(
        {
            "status": "completed",
            "generation": GENERATIONS - 1,
            "generations_total": GENERATIONS,
            "best_airfoil": best["airfoil"],
            "best_span": best["wing_span"],
            "best_area": best["wing_area"],
            "best_velocity": best["velocity"],
            "best_battery_wh": best["battery_wh"],
            "best_dynamic_pressure": best.get("dynamic_pressure"),
            "best_feasible": feasible,
            "best_lift": best["lift"],
            "best_weight": best_weight,
            "best_lift_margin": lift_margin,
            "best_drag": best["drag"],
            "best_power": best_power_val if best_power_val is not None else (best.get("drag") * best.get("velocity") if best.get("drag") is not None and best.get("velocity") is not None else None),
            "battery_capacity_Wh": best["battery_wh"],
            "best_battery_weight": best["battery_weight"],
            "best_battery_energy_J": best["battery_energy_j"],
            "best_flight_time_s": best_flight_time_s,
            "best_flight_time_min": best_flight_time_min,
            "best_ld": best["raw_score"],
                "best_mission_fitness": best["mission_fitness"],
                "best_time_score": best["time_score"],
                "best_adjusted_fitness": best["adjusted_score"],
                "pareto_front": pareto_state(pareto_front),
                "pareto_front_count": len(pareto_front),
                "mission_payload": MISSION["payload_weight"],
                "mission_time": MISSION["target_flight_time"],
                "mission_speed": MISSION["target_speed"],
            "weight_target": WEIGHT,
            "dynamic_pressure": best.get("dynamic_pressure"),
            "best_history": best_history,
            "population": population_state(scored_population),
            "source_counts": source_counts,
            "xfoil_calls": xfoil_calls,
            "ml_predictions": ml_predictions,
            "ml_skips": ml_skips,
            "runtime_seconds": runtime,
        }
    )

    with open("experiment_summary.txt", "w") as f:
        f.write(f"Mission Payload (N): {MISSION['payload_weight']}\n")
        f.write(f"Mission Target Flight Time (min): {MISSION['target_flight_time']}\n")
        f.write(f"Mission Target Speed (m/s): {MISSION['target_speed']}\n")
        f.write(f"Use Surrogate: {USE_SURROGATE}\n")
        f.write(f"Weight Target: {WEIGHT}\n")
        f.write(f"Best Velocity: {best['velocity']}\n")
        f.write(f"Best Dynamic Pressure: {best_dynamic_pressure}\n")
        f.write(f"Baseline Wing: {format_design_label(baseline_design)}\n")
        f.write(f"Baseline L/D: {baseline_result['score']}\n")
        f.write(f"Baseline Lift: {baseline_result['lift']}\n")
        f.write(f"Best Airfoil: {best['airfoil']}\n")
        f.write(f"Best Wing Span: {best['wing_span']}\n")
        f.write(f"Best Wing Area: {best['wing_area']}\n")
        f.write(f"Best Battery Wh: {best['battery_wh']}\n")
        f.write(f"Best Weight: {best_weight}\n")
        f.write(f"Best Flight Time Min: {best_flight_time_min}\n")
        f.write(f"Best Mission Fitness: {best['mission_fitness']}\n")
        f.write(f"Best Adjusted Fitness: {best['adjusted_score']}\n")
        f.write(f"Pareto Front Size: {len(pareto_front)}\n")
        f.write(f"Best Lift: {best['lift']}\n")
        f.write(f"Best L/D: {best['raw_score']}\n")
        f.write(f"XFOIL calls: {xfoil_calls}\n")
        f.write(f"ML predictions: {ml_predictions}\n")
        f.write(f"ML skips: {ml_skips}\n")
        f.write(f"Cache size: {len(fitness_cache)}\n")
        f.write(f"Runtime: {runtime}\n")
