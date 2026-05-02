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
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

from airfoil_generator import generate_random_naca
from airfoil_plotter import plot_airfoil
from rl_agent import RLAgent
from surrogate_model import predict_with_uncertainty, train_model
from xfoil_runner import run_xfoil


POPULATION_SIZE = 30
GENERATIONS = 15
MUTATION_RATE = 0.6
CACHE_FILE = Path("airfoil_cache.json")
USE_SURROGATE = True
RL_ENABLED = True
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
AIR_VISCOSITY = 1.81e-5
RE_MIN = 50000
RE_MAX = 300000
STRUCTURE_WEIGHT = 4.0
BATTERY_WEIGHT_PER_WH = 0.02
BATTERY_MIN_WH = 100.0
BATTERY_MAX_WH = 500.0
DEFAULT_BATTERY_WH = 200.0
DEFAULT_XFOIL_REYNOLDS = 1000000.0
DEFAULT_XFOIL_ALPHA = 5.0
AIRFOIL_ALPHA_SWEEP = (-2.0, 0.0, 2.0, 4.0, 6.0, 8.0)
REYNOLDS_CACHE_BIN = 1000
WEIGHT = MISSION["payload_weight"] + STRUCTURE_WEIGHT + BATTERY_WEIGHT_PER_WH * DEFAULT_BATTERY_WH
OSWALD_EFFICIENCY = 0.85
UNCERTAINTY_THRESHOLD = 5.0
ML_GUIDED_SEARCH = True
ML_GUIDED_CANDIDATES = 5
ML_GUIDED_UNCERTAINTY_THRESHOLD = 15.0
ML_GUIDED_MIN_PREDICTED_GAIN = 0.5
RL_WARMUP_GENERATIONS = 5
RL_SELECTION_PROBABILITY = 0.3
FIXED_AIRFOIL = None
_FIXED_AIRFOIL_UNSET = object()

best_designs = []
xfoil_calls = 0
ml_skips = 0
ml_predictions = 0
ml_guided_attempts = 0
ml_guided_accepts = 0
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


def compute_chord(wing_area, wing_span):
    if wing_span <= 0:
        return None
    return wing_area / wing_span


def compute_reynolds(velocity, chord):
    if chord is None or velocity is None:
        return None
    return (RHO * velocity * chord) / AIR_VISCOSITY


def reynolds_cache_bucket(reynolds):
    if reynolds is None:
        return None
    return int(reynolds // REYNOLDS_CACHE_BIN) * REYNOLDS_CACHE_BIN


def is_valid_reynolds(reynolds):
    return reynolds is not None and RE_MIN <= reynolds <= RE_MAX


def make_airfoil_cache_key(naca, reynolds=None, alpha=DEFAULT_XFOIL_ALPHA):
    if reynolds is None:
        return naca

    re_key = reynolds_cache_bucket(reynolds)
    if isinstance(alpha, str):
        alpha_key = alpha
    else:
        alpha_key = f"{float(alpha):.2f}"
    return f"{naca}|Re={re_key}|alpha={alpha_key}"


def make_airfoil_sweep_cache_key(naca, reynolds=None, alpha_values=None):
    if reynolds is None:
        return naca

    values = AIRFOIL_ALPHA_SWEEP if alpha_values is None else tuple(alpha_values)
    sweep_key = "sweep[" + ",".join(f"{float(alpha):.2f}" for alpha in values) + "]"
    return make_airfoil_cache_key(naca, reynolds, sweep_key)


def compute_lift_drag_ratio(cl, cd):
    if cl is None or cd in (None, 0):
        return 0

    ld = cl / cd

    if cl < 0 or cl > 2.0:
        ld *= 0.2

    if cd < 0.003 or cd > 0.05:
        ld *= 0.2

    if ld > 250:
        ld *= 0.2

    return ld


def velocity_bounds_for_re(span, area, re_min=RE_MIN, re_max=RE_MAX):
    chord = compute_chord(area, span)
    if chord is None or chord <= 0:
        return None, None

    v_min = (re_min * AIR_VISCOSITY) / (RHO * chord)
    v_max = (re_max * AIR_VISCOSITY) / (RHO * chord)
    return v_min, v_max


def repair_design_for_re(design, target_speed=None):
    target_speed = MISSION["target_speed"] if target_speed is None else target_speed
    repaired = clone_design(design)

    span = repaired["wing_span"]
    area = repaired["wing_area"]
    velocity = repaired["velocity"]

    mission_v_min = max(VELOCITY_MIN, target_speed * 0.8)
    mission_v_max = min(VELOCITY_MAX, target_speed * 1.2)

    chord = compute_chord(area, span)
    if chord is None or chord <= 0:
        return repaired

    re_chord_min = (RE_MIN * AIR_VISCOSITY) / (RHO * mission_v_max)
    re_chord_max = (RE_MAX * AIR_VISCOSITY) / (RHO * mission_v_min)

    if chord < re_chord_min or chord > re_chord_max:
        target_chord = clamp(chord, re_chord_min, re_chord_max)
        span_min_for_chord = max(SPAN_MIN, AREA_MIN / target_chord)
        span_max_for_chord = min(SPAN_MAX, AREA_MAX / target_chord)

        if span_min_for_chord <= span_max_for_chord:
            span = clamp(span, span_min_for_chord, span_max_for_chord)
            area = target_chord * span
        else:
            span = clamp(span, SPAN_MIN, SPAN_MAX)
            area = clamp(target_chord * span, AREA_MIN, AREA_MAX)

    v_min, v_max = velocity_bounds_for_re(span, area)
    if v_min is None or v_max is None:
        return create_design(repaired["airfoil"], span, area, velocity, repaired["battery_wh"])

    v_lo = max(v_min, mission_v_min)
    v_hi = min(v_max, mission_v_max)

    if v_lo < v_hi:
        velocity = random.uniform(v_lo, v_hi)
    else:
        velocity = clamp(target_speed, v_min, v_max)

    final_design = create_design(repaired["airfoil"], span, area, velocity, repaired["battery_wh"])
    final_chord = compute_chord(final_design["wing_area"], final_design["wing_span"])
    final_reynolds = compute_reynolds(final_design["velocity"], final_chord)

    if not is_valid_reynolds(final_reynolds):
        final_v_min, final_v_max = velocity_bounds_for_re(final_design["wing_span"], final_design["wing_area"])
        if final_v_min is not None and final_v_max is not None:
            if final_reynolds > RE_MAX:
                target_velocity = min(final_v_max, mission_v_max) * 0.995
            else:
                target_velocity = max(final_v_min, mission_v_min) * 1.005

            final_design["velocity"] = round_velocity(clamp(target_velocity, final_v_min, final_v_max))

    return final_design


def battery_weight(battery_wh):
    return BATTERY_WEIGHT_PER_WH * battery_wh


def design_weight(battery_wh):
    return MISSION["payload_weight"] + STRUCTURE_WEIGHT + battery_weight(battery_wh)


def estimate_lift(design, cl=0.8):
    velocity = design["velocity"]
    area = design["wing_area"]
    return 0.5 * RHO * velocity ** 2 * area * cl


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
    design = create_design(
        FIXED_AIRFOIL if FIXED_AIRFOIL else generate_random_naca(),
        random.uniform(SPAN_MIN, SPAN_MAX),
        random.uniform(AREA_MIN, AREA_MAX),
        random.uniform(target_speed * 0.8, target_speed * 1.2),
        random.uniform(BATTERY_MIN_WH, BATTERY_MAX_WH),
    )
    return repair_design_for_re(design, target_speed)


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

    if FIXED_AIRFOIL:
        mutated["airfoil"] = FIXED_AIRFOIL
    elif random.random() < 0.65:
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

    return repair_design_for_re(mutated)


def perturb_design(design):
    perturbed = clone_design(design)

    if FIXED_AIRFOIL:
        perturbed["airfoil"] = FIXED_AIRFOIL
    elif random.random() < 0.8:
        perturbed["airfoil"] = mutate_airfoil(perturbed["airfoil"])

    perturbed["wing_span"] = round_design_value(
        clamp(perturbed["wing_span"] + random.uniform(-0.12, 0.12), SPAN_MIN, SPAN_MAX)
    )
    perturbed["wing_area"] = round_design_value(
        clamp(perturbed["wing_area"] + random.uniform(-0.05, 0.05), AREA_MIN, AREA_MAX)
    )
    perturbed["velocity"] = round_velocity(
        clamp_velocity(perturbed["velocity"] + random.uniform(-0.9, 0.9))
    )
    perturbed["battery_wh"] = round_battery_wh(
        clamp_battery_wh(perturbed["battery_wh"] + random.uniform(-20.0, 20.0))
    )

    return repair_design_for_re(perturbed)


def crossover_airfoils(parent1, parent2):
    d1 = parent1.replace("NACA ", "")
    d2 = parent2.replace("NACA ", "")
    child_digits = ""

    for idx in range(4):
        child_digits += random.choice([d1[idx], d2[idx]])

    return "NACA " + child_digits


def crossover_design(parent1, parent2):
    child = create_design(
        FIXED_AIRFOIL if FIXED_AIRFOIL else crossover_airfoils(parent1["airfoil"], parent2["airfoil"]),
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
    return repair_design_for_re(child)


def evaluate_design_candidate(design, model=None):
    chord = compute_chord(design["wing_area"], design["wing_span"])
    reynolds = compute_reynolds(design["velocity"], chord)

    if not is_valid_reynolds(reynolds):
        return {
            "design": clone_design(design),
            "airfoil_details": {
                "score": 0,
                "cl": None,
                "cd": None,
                "evaluation_type": "invalid",
                "surrogate_used": False,
                "surrogate_mean_ld": None,
                "surrogate_uncertainty": None,
            },
            "fitness": 0,
            "raw_score": 0,
            "mission_fitness": 0,
            "score": 0,
            "time_score": 0,
            "lift": 0,
            "drag": 0,
            "power": None,
            "weight": design_weight(design.get("battery_wh", DEFAULT_BATTERY_WH)),
            "battery_weight": battery_weight(design.get("battery_wh", DEFAULT_BATTERY_WH)),
            "battery_energy_j": battery_energy_j(design.get("battery_wh", DEFAULT_BATTERY_WH)),
            "flight_time_s": None,
            "flight_time_min": None,
            "lift_margin": None,
            "constraint_satisfied": False,
            "aspect_ratio": design["wing_span"] ** 2 / design["wing_area"] if design["wing_area"] > 0 else None,
            "base_ld": 0,
            "base_cd": None,
            "total_cd": None,
            "dynamic_pressure": None,
            "chord": chord,
            "reynolds": reynolds,
            "reynolds_valid": False,
            "evaluation_type": "invalid",
            "surrogate_used": False,
            "surrogate_mean_ld": None,
            "surrogate_uncertainty": None,
            "reason": "invalid_reynolds",
        }

    airfoil_details = evaluate_airfoil_details(
        design["airfoil"],
        model,
        reynolds=reynolds,
    )
    design_score = score_design(design, airfoil_details, chord=chord, reynolds=reynolds)

    return {
        "design": clone_design(design),
        "airfoil_details": airfoil_details,
        "fitness": design_score["mission_fitness"],
        "raw_score": design_score["score"],
        "mission_fitness": design_score["mission_fitness"],
        "score": design_score["score"],
        "time_score": design_score["time_score"],
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
        "chord": chord,
        "reynolds": reynolds,
        "reynolds_valid": True,
        "evaluation_type": airfoil_details["evaluation_type"],
        "surrogate_used": airfoil_details["surrogate_used"],
        "surrogate_mean_ld": airfoil_details["surrogate_mean_ld"],
        "surrogate_uncertainty": airfoil_details["surrogate_uncertainty"],
    }


def try_ml_improvement(design, model=None, baseline_result=None):
    if model is None:
        return None

    baseline = baseline_result if baseline_result is not None else evaluate_design_candidate(design, model)
    baseline_pred = (
        baseline["surrogate_mean_ld"]
        if baseline["surrogate_mean_ld"] is not None
        else baseline["raw_score"]
    )

    shortlisted = []
    for _ in range(ML_GUIDED_CANDIDATES):
        candidate = perturb_design(design)
        predicted_ld, uncertainty = predict_with_uncertainty(model, candidate)
        if uncertainty is None or uncertainty >= ML_GUIDED_UNCERTAINTY_THRESHOLD:
            continue
        shortlisted.append((candidate, predicted_ld, uncertainty))

    if not shortlisted:
        return baseline

    best_candidate, best_predicted_ld, _ = max(shortlisted, key=lambda item: item[1])
    if best_predicted_ld <= baseline_pred + ML_GUIDED_MIN_PREDICTED_GAIN:
        return baseline

    candidate_result = evaluate_design_candidate(best_candidate, model)
    if candidate_result["fitness"] > baseline["fitness"]:
        return candidate_result

    return baseline


def get_cached_airfoil_entry(naca):
    entry = fitness_cache.get(naca)
    if isinstance(entry, dict):
        return {
            "cl": entry.get("cl"),
            "cd": entry.get("cd"),
            "ld": entry.get("ld", 0),
            "evaluation_type": entry.get("evaluation_type"),
            "surrogate_mean_ld": entry.get("surrogate_mean_ld"),
            "surrogate_uncertainty": entry.get("surrogate_uncertainty"),
            "reynolds": entry.get("reynolds"),
            "alpha": entry.get("alpha"),
        }

    return None


def evaluate_airfoil_details(naca, model=None, reynolds=None, alpha_sweep=AIRFOIL_ALPHA_SWEEP):
    global ml_predictions, xfoil_calls, ml_skips

    original_reynolds = reynolds
    requested_reynolds = DEFAULT_XFOIL_REYNOLDS if reynolds is None else reynolds
    alpha_values = tuple(alpha_sweep) if alpha_sweep is not None else (DEFAULT_XFOIL_ALPHA,)
    if len(alpha_values) == 0:
        alpha_values = (DEFAULT_XFOIL_ALPHA,)

    cache_key = make_airfoil_sweep_cache_key(naca, requested_reynolds, alpha_values)

    cached_entry = get_cached_airfoil_entry(cache_key)
    if cached_entry is None and original_reynolds is None:
        cached_entry = get_cached_airfoil_entry(naca)

    if cached_entry is not None and cached_entry["evaluation_type"] in {"ml", "skipped"}:
        return {
            "score": cached_entry["ld"],
            "cl": None,
            "cd": None,
            "evaluation_type": "skipped",
            "surrogate_used": True,
            "surrogate_mean_ld": cached_entry["surrogate_mean_ld"],
            "surrogate_uncertainty": cached_entry["surrogate_uncertainty"],
        }

    if cached_entry is not None and cached_entry["cl"] is not None and cached_entry["cd"] is not None:
        return {
            "score": cached_entry["ld"],
            "cl": cached_entry["cl"],
            "cd": cached_entry["cd"],
            "evaluation_type": "cached",
            "surrogate_used": False,
            "surrogate_mean_ld": None,
            "surrogate_uncertainty": None,
            "alpha": cached_entry.get("alpha"),
        }

    surrogate_mean_ld = None
    surrogate_uncertainty = None
    surrogate_used = False
    if model is not None:
        with counter_lock:
            ml_predictions += 1
        surrogate_used = True
        surrogate_mean_ld, surrogate_uncertainty = predict_with_uncertainty(model, naca)
        if surrogate_mean_ld is not None and surrogate_uncertainty is not None and surrogate_uncertainty < UNCERTAINTY_THRESHOLD:
            with counter_lock:
                ml_skips += 1

            cached_skip = {
                "cl": None,
                "cd": None,
                "ld": float(surrogate_mean_ld),
                "reynolds": requested_reynolds,
                "alpha": None,
                "alpha_samples": list(alpha_values),
                "best_alpha": None,
                "evaluation_type": "skipped",
                "surrogate_mean_ld": float(surrogate_mean_ld),
                "surrogate_uncertainty": float(surrogate_uncertainty),
            }
            fitness_cache[cache_key] = cached_skip
            if original_reynolds is None:
                fitness_cache[naca] = cached_skip
            return {
                "score": float(surrogate_mean_ld),
                "cl": None,
                "cd": None,
                "evaluation_type": "skipped",
                "surrogate_used": True,
                "surrogate_mean_ld": float(surrogate_mean_ld),
                "surrogate_uncertainty": float(surrogate_uncertainty),
                "alpha": None,
            }

    sweep_results = []
    for alpha in alpha_values:
        alpha_cache_key = make_airfoil_cache_key(naca, requested_reynolds, alpha)
        cached_alpha_entry = get_cached_airfoil_entry(alpha_cache_key)

        if (
            cached_alpha_entry is not None
            and cached_alpha_entry.get("cl") is not None
            and cached_alpha_entry.get("cd") is not None
        ):
            cl = cached_alpha_entry["cl"]
            cd = cached_alpha_entry["cd"]
            ld = cached_alpha_entry["ld"]
        else:
            with counter_lock:
                xfoil_calls += 1
            cl, cd = run_xfoil(naca, reynolds=requested_reynolds, alpha=alpha)
            ld = compute_lift_drag_ratio(cl, cd)
            fitness_cache[alpha_cache_key] = {
                "cl": 0 if cl is None else cl,
                "cd": cd,
                "ld": ld,
                "reynolds": requested_reynolds,
                "alpha": alpha,
                "evaluation_type": "simulated",
            }

        sweep_results.append(
            {
                "alpha": alpha,
                "cl": cl,
                "cd": cd,
                "ld": ld,
                "source": "cached" if cached_alpha_entry is not None and cached_alpha_entry.get("cl") is not None and cached_alpha_entry.get("cd") is not None else "simulated",
            }
        )

    valid_results = [result for result in sweep_results if result["cd"] not in (None, 0)]
    best_result = max(valid_results or sweep_results, key=lambda result: result["ld"], default=None)

    if best_result is None:
        summary = {
            "cl": 0,
            "cd": None,
            "ld": 0,
            "reynolds": requested_reynolds,
            "alpha": None,
            "alpha_samples": list(alpha_values),
            "best_alpha": None,
            "best_ld": 0,
            "sweep_results": sweep_results,
            "evaluation_type": "simulated",
        }
        fitness_cache[cache_key] = summary
        if original_reynolds is None:
            fitness_cache[naca] = summary
        return {
            "score": 0,
            "cl": 0,
            "cd": None,
            "evaluation_type": "simulated",
            "surrogate_used": surrogate_used,
            "surrogate_mean_ld": surrogate_mean_ld,
            "surrogate_uncertainty": surrogate_uncertainty,
            "alpha": None,
            "alpha_samples": list(alpha_values),
            "best_alpha": None,
            "sweep_results": sweep_results,
        }

    summary = {
        "cl": best_result["cl"],
        "cd": best_result["cd"],
        "ld": best_result["ld"],
        "reynolds": requested_reynolds,
        "alpha": best_result["alpha"],
        "alpha_samples": list(alpha_values),
        "best_alpha": best_result["alpha"],
        "best_ld": best_result["ld"],
        "sweep_results": sweep_results,
        "evaluation_type": "simulated",
    }
    fitness_cache[cache_key] = summary
    if original_reynolds is None:
        fitness_cache[naca] = summary
    return {
        "score": best_result["ld"],
        "cl": best_result["cl"],
        "cd": best_result["cd"],
        "evaluation_type": "simulated",
        "surrogate_used": surrogate_used,
        "surrogate_mean_ld": surrogate_mean_ld,
        "surrogate_uncertainty": surrogate_uncertainty,
        "alpha": best_result["alpha"],
        "alpha_samples": list(alpha_values),
        "best_alpha": best_result["alpha"],
        "sweep_results": sweep_results,
    }


def evaluate_airfoil(naca, model=None):
    return evaluate_airfoil_details(naca, model)["score"]


def induced_drag_coefficient(cl, aspect_ratio):
    if aspect_ratio <= 0:
        return 1.0

    return (cl ** 2) / (math.pi * OSWALD_EFFICIENCY * aspect_ratio)


def score_design(design, airfoil_details, chord=None, reynolds=None):
    wing_area = design["wing_area"]
    wing_span = design["wing_span"]
    battery_wh = design["battery_wh"]
    weight = design_weight(battery_wh)
    aspect_ratio = wing_span ** 2 / wing_area
    chord = chord if chord is not None else compute_chord(wing_area, wing_span)
    reynolds = reynolds if reynolds is not None else compute_reynolds(design["velocity"], chord)
    cl = airfoil_details.get("cl") or 0
    base_cd = airfoil_details.get("cd")
    base_ld = airfoil_details.get("score", 0)
    target_time = MISSION["target_flight_time"]

    if base_cd in (None, 0) or cl <= 0:
        if airfoil_details.get("evaluation_type") in {"skipped", "ml"} and base_ld is not None:
            approx_lift = estimate_lift(design)
            constraint_satisfied = approx_lift >= weight
            surrogate_score = float(base_ld)
            if not constraint_satisfied:
                return {
                    "score": 0,
                    "mission_fitness": 0,
                    "time_score": 0,
                    "lift": approx_lift,
                    "drag": 0,
                    "power": None,
                    "weight": weight,
                    "battery_weight": battery_weight(battery_wh),
                    "battery_energy_j": battery_energy_j(battery_wh),
                    "flight_time_s": None,
                    "flight_time_min": None,
                    "lift_margin": approx_lift - weight,
                    "constraint_satisfied": False,
                    "aspect_ratio": aspect_ratio,
                    "base_ld": 0,
                    "base_cd": None,
                    "total_cd": None,
                    "chord": chord,
                    "reynolds": reynolds,
                    "realism_penalty": 1.0,
                }

            return {
                "score": surrogate_score,
                "mission_fitness": surrogate_score,
                "time_score": 0,
                "lift": approx_lift,
                "drag": 0,
                "power": None,
                "weight": weight,
                "battery_weight": battery_weight(battery_wh),
                "battery_energy_j": battery_energy_j(battery_wh),
                "flight_time_s": None,
                "flight_time_min": None,
                "lift_margin": approx_lift - weight,
                "constraint_satisfied": True,
                "aspect_ratio": aspect_ratio,
                "base_ld": surrogate_score,
                "base_cd": None,
                "total_cd": None,
                "chord": chord,
                "reynolds": reynolds,
                "realism_penalty": 1.0,
            }

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
            "chord": chord,
            "reynolds": reynolds,
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
        "chord": chord,
        "reynolds": reynolds,
        "realism_penalty": realism_penalty,
    }


def evaluate_design_simple(design, model=None):
    airfoil_details = evaluate_airfoil_details(design["airfoil"], model)
    result = score_design(design, airfoil_details)
    return result["mission_fitness"]


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
                "reynolds": design_score["reynolds"],
                "reynolds_valid": is_valid_reynolds(design_score["reynolds"]),
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
            "reynolds": entry["reynolds"],
            "reynolds_valid": entry["reynolds_valid"],
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


def run_ga(
    use_surrogate=None,
    guided_search=None,
    rl_enabled=None,
    seed=None,
    fixed_airfoil=_FIXED_AIRFOIL_UNSET,
    return_metrics=False,
):
    global xfoil_calls, ml_predictions, ml_skips, ml_guided_attempts, ml_guided_accepts
    global USE_SURROGATE, ML_GUIDED_SEARCH, RL_ENABLED, FIXED_AIRFOIL

    start_time = time.time()
    if seed is not None:
        random.seed(seed)

    original_use_surrogate = USE_SURROGATE
    original_guided_search = ML_GUIDED_SEARCH
    original_rl_enabled = RL_ENABLED
    original_fixed_airfoil = FIXED_AIRFOIL

    if use_surrogate is not None:
        USE_SURROGATE = use_surrogate
    if guided_search is not None:
        ML_GUIDED_SEARCH = guided_search
    if rl_enabled is not None:
        RL_ENABLED = rl_enabled
    if fixed_airfoil is not _FIXED_AIRFOIL_UNSET:
        FIXED_AIRFOIL = fixed_airfoil

    best_designs.clear()
    xfoil_calls = 0
    ml_predictions = 0
    ml_skips = 0
    ml_guided_attempts = 0
    ml_guided_accepts = 0
    rl_used = 0
    rl_rewards = []
    rl_improvements = 0
    rl_total_steps = 0
    apply_mission(load_mission())
    if not CONTROL_FILE.exists():
        with open(CONTROL_FILE, "w") as f:
            json.dump({"running": True, "mission": dict(MISSION)}, f, indent=2)
    model = train_model() if USE_SURROGATE else None
    rl_agent = RLAgent() if RL_ENABLED else None
    population = [generate_random_design() for _ in range(POPULATION_SIZE)]
    best_history = []
    stats = []
    re_values = []

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
            "best_reynolds": None,
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
            "ml_replacements": ml_skips,
            "ml_replacement_rate": 0.0,
            "xfoil_reduction_count": ml_skips,
            "xfoil_reduction_pct": 0.0,
            "ml_guided_attempts": ml_guided_attempts,
            "ml_guided_accepts": ml_guided_accepts,
            "ml_guided_acceptance_pct": 0.0,
            "generation_xfoil_calls": 0,
            "generation_predictions": 0,
            "generation_ml_skips": 0,
            "generation_ml_guided_attempts": 0,
            "generation_ml_guided_accepts": 0,
            "rl_total_steps": 0,
            "rl_improvements": 0,
            "rl_improvement_rate": 0.0,
            "rl_average_reward": 0.0,
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
        generation_guided_attempts_before = ml_guided_attempts
        generation_guided_accepts_before = ml_guided_accepts

        scored_population = evaluate_population(population, model)
        pareto_front = build_pareto_front(scored_population)

        for entry in scored_population:
            if entry["reynolds"] is not None:
                re_values.append(entry["reynolds"])
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
                best["reynolds"],
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
                "best_reynolds": best["reynolds"],
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
                "xfoil_reduction_count": ml_skips,
                "xfoil_reduction_pct": (ml_skips / ml_predictions * 100.0) if ml_predictions else 0.0,
                "ml_guided_attempts": ml_guided_attempts,
                "ml_guided_accepts": ml_guided_accepts,
                "ml_guided_acceptance_pct": (ml_guided_accepts / ml_guided_attempts * 100.0) if ml_guided_attempts else 0.0,
                "generation_xfoil_calls": xfoil_calls - generation_xfoil_before,
                "generation_predictions": ml_predictions - generation_predictions_before,
                "generation_ml_skips": ml_skips - generation_skips_before,
                "generation_ml_guided_attempts": ml_guided_attempts - generation_guided_attempts_before,
                "generation_ml_guided_accepts": ml_guided_accepts - generation_guided_accepts_before,
                "rl_total_steps": rl_total_steps,
                "rl_improvements": rl_improvements,
                "rl_improvement_rate": (rl_improvements / rl_total_steps) if rl_total_steps else 0.0,
                "rl_average_reward": (sum(rl_rewards) / len(rl_rewards)) if rl_rewards else 0.0,
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
                if RL_ENABLED and generation >= RL_WARMUP_GENERATIONS and random.random() < RL_SELECTION_PROBABILITY:
                    base_child = child
                    child, reward = rl_agent.improve(
                        child,
                        lambda d: evaluate_design_candidate(d, model)["mission_fitness"],
                        return_reward=True,
                    )
                    rl_rewards.append(reward)
                    rl_total_steps += 1
                    if reward > 0:
                        rl_improvements += 1
                        child = create_design(
                            child["airfoil"],
                            child["wing_span"],
                            child["wing_area"],
                            child["velocity"],
                            child["battery_wh"],
                        )
                        rl_used += 1
                    else:
                        child = mutate_design(base_child)
                else:
                    child = mutate_design(child)

            child_result = None
            if USE_SURROGATE and ML_GUIDED_SEARCH and model is not None:
                child_result = evaluate_design_candidate(child, model)
                with counter_lock:
                    ml_guided_attempts += 1
                improved = try_ml_improvement(child, model, child_result)
                if improved is not None and improved["fitness"] > child_result["fitness"]:
                    with counter_lock:
                        ml_guided_accepts += 1
                    child = improved["design"]

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

    valid_re_values = [value for value in re_values if value is not None]
    if valid_re_values:
        re_plot_path = Path("reynolds_distribution.png")
        plt.figure()
        plt.hist(valid_re_values, bins=20)
        plt.title("Reynolds Number Distribution")
        plt.xlabel("Re")
        plt.ylabel("Count")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.savefig(re_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved Reynolds distribution plot to", re_plot_path)

    if rl_rewards:
        plt.figure()
        plt.plot(rl_rewards, alpha=0.7)
        plt.title("RL Reward Over Time")
        plt.xlabel("RL Step")
        plt.ylabel("Reward")
        plt.grid(True)
        rl_plot_path = Path("rl_learning_curve.png")
        plt.savefig(rl_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved RL learning curve to", rl_plot_path)

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
                "reynolds",
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
    baseline_chord = compute_chord(baseline_design["wing_area"], baseline_design["wing_span"])
    baseline_reynolds = compute_reynolds(baseline_design["velocity"], baseline_chord)
    baseline_airfoil_details = evaluate_airfoil_details(
        baseline_design["airfoil"],
        None,
        reynolds=baseline_reynolds,
    )
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
    if valid_re_values:
        print("Reynolds min:", min(valid_re_values))
        print("Reynolds max:", max(valid_re_values))
        print("Reynolds mean:", sum(valid_re_values) / len(valid_re_values))
    print("\n--- Optimization Statistics ---")
    print("Total XFOIL simulations:", xfoil_calls)
    print("ML predictions:", ml_predictions)
    print("ML skipped designs:", ml_skips)
    print("ML-guided search attempts:", ml_guided_attempts)
    print("ML-guided search acceptances:", ml_guided_accepts)
    print("XFOIL calls reduced by ML:", ml_skips)
    print("ML skip rate (%):", round((ml_skips / ml_predictions * 100.0) if ml_predictions else 0.0, 2))
    print(
        "ML guided acceptance rate (%):",
        round((ml_guided_accepts / ml_guided_attempts * 100.0) if ml_guided_attempts else 0.0, 2),
    )
    print("RL improvements used:", rl_used)
    if rl_total_steps > 0:
        improvement_rate = rl_improvements / rl_total_steps
        avg_reward = sum(rl_rewards) / len(rl_rewards)
        print("\n--- RL Performance ---")
        print("RL steps:", rl_total_steps)
        print("Improvement rate:", round(improvement_rate, 3))
        print("Average reward:", round(avg_reward, 3))
    print("ML replacements:", ml_skips)
    print(
        "Reduction (%):",
        round((ml_skips / (ml_skips + xfoil_calls) * 100.0) if (ml_skips + xfoil_calls) else 0.0, 2),
    )
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
                "best_reynolds": best["reynolds"],
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
            "ml_replacements": ml_skips,
            "ml_replacement_rate": (ml_skips / (ml_skips + xfoil_calls)) if (ml_skips + xfoil_calls) else 0.0,
            "xfoil_reduction_count": ml_skips,
            "xfoil_reduction_pct": (ml_skips / ml_predictions * 100.0) if ml_predictions else 0.0,
            "rl_improvements_used": rl_used,
            "rl_total_steps": rl_total_steps,
            "rl_improvements": rl_improvements,
            "rl_improvement_rate": (rl_improvements / rl_total_steps) if rl_total_steps else 0.0,
            "rl_average_reward": (sum(rl_rewards) / len(rl_rewards)) if rl_rewards else 0.0,
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
        f.write(f"Best Reynolds: {best['reynolds']}\n")
        f.write(f"Pareto Front Size: {len(pareto_front)}\n")
        f.write(f"Best Lift: {best['lift']}\n")
        f.write(f"Best L/D: {best['raw_score']}\n")
        if valid_re_values:
            f.write(f"Reynolds Min: {min(valid_re_values)}\n")
            f.write(f"Reynolds Max: {max(valid_re_values)}\n")
            f.write(f"Reynolds Mean: {sum(valid_re_values) / len(valid_re_values)}\n")
        f.write(f"XFOIL calls: {xfoil_calls}\n")
        f.write(f"ML predictions: {ml_predictions}\n")
        f.write(f"ML skips: {ml_skips}\n")
        f.write(f"ML replacements: {ml_skips}\n")
        f.write(f"ML-guided search attempts: {ml_guided_attempts}\n")
        f.write(f"ML-guided search acceptances: {ml_guided_accepts}\n")
        f.write(f"XFOIL calls reduced by ML: {ml_skips}\n")
        f.write(f"Reduction rate (%): {((ml_skips / (ml_skips + xfoil_calls)) * 100.0) if (ml_skips + xfoil_calls) else 0.0}\n")
        f.write(f"ML skip rate (%): {((ml_skips / ml_predictions) * 100.0) if ml_predictions else 0.0}\n")
        f.write(
            f"ML guided acceptance rate (%): {((ml_guided_accepts / ml_guided_attempts) * 100.0) if ml_guided_attempts else 0.0}\n"
        )
        f.write(f"RL improvements used: {rl_used}\n")
        f.write(f"RL total steps: {rl_total_steps}\n")
        f.write(f"RL improvements: {rl_improvements}\n")
        f.write(f"RL improvement rate: {(rl_improvements / rl_total_steps) if rl_total_steps else 0.0}\n")
        f.write(f"RL average reward: {(sum(rl_rewards) / len(rl_rewards)) if rl_rewards else 0.0}\n")
        f.write(f"Cache size: {len(fitness_cache)}\n")
        f.write(f"Runtime: {runtime}\n")
    
    result = {
        "best": best,
        "best_history": best_history,
        "runtime_seconds": runtime,
        "xfoil_calls": xfoil_calls,
        "ml_predictions": ml_predictions,
        "ml_skips": ml_skips,
        "ml_replacements": ml_skips,
        "ml_replacement_rate": (ml_skips / (ml_skips + xfoil_calls)) if (ml_skips + xfoil_calls) else 0.0,
        "xfoil_reduction_count": ml_skips,
        "xfoil_reduction_pct": (ml_skips / ml_predictions * 100.0) if ml_predictions else 0.0,
        "ml_guided_attempts": ml_guided_attempts,
        "ml_guided_accepts": ml_guided_accepts,
        "ml_guided_acceptance_pct": (ml_guided_accepts / ml_guided_attempts * 100.0) if ml_guided_attempts else 0.0,
        "rl_used": rl_used,
        "rl_total_steps": rl_total_steps,
        "rl_improvements": rl_improvements,
        "rl_improvement_rate": (rl_improvements / rl_total_steps) if rl_total_steps else 0.0,
        "rl_average_reward": (sum(rl_rewards) / len(rl_rewards)) if rl_rewards else 0.0,
        "rl_rewards": rl_rewards,
        "best_flight_time_min": best_flight_time_min,
        "baseline_result": baseline_result,
        "pareto_front_size": len(pareto_front),
        "re_min": min(valid_re_values) if valid_re_values else None,
        "re_max": max(valid_re_values) if valid_re_values else None,
        "re_mean": (sum(valid_re_values) / len(valid_re_values)) if valid_re_values else None,
    }

    USE_SURROGATE = original_use_surrogate
    ML_GUIDED_SEARCH = original_guided_search
    RL_ENABLED = original_rl_enabled
    FIXED_AIRFOIL = original_fixed_airfoil

    if return_metrics:
        return result

    return best, best_history

if __name__ == "__main__":
    run_ga()
