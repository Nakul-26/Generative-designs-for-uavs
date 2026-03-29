import json
from pathlib import Path

import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    RandomForestRegressor = None


CACHE_FILE = Path("airfoil_cache.json")


def parse_naca(naca):
    digits = naca.replace("NACA ", "")

    camber = int(digits[0])
    position = int(digits[1])
    thickness = int(digits[2:])

    return [camber, position, thickness]


def load_dataset():
    if not CACHE_FILE.exists():
        return np.array([]), np.array([])

    with open(CACHE_FILE, "r") as f:
        data = json.load(f)

    X = []
    y = []

    for airfoil, entry in data.items():
        ld = entry["ld"] if isinstance(entry, dict) else entry
        features = parse_naca(airfoil)
        X.append(features)
        y.append(ld)

    return np.array(X), np.array(y)


def train_model():
    if RandomForestRegressor is None:
        print(
            "scikit-learn is not installed. Running without the surrogate model. "
            "Install it with `pip install scikit-learn` to enable ML predictions."
        )
        return None

    X, y = load_dataset()

    if len(X) < 10:
        return None

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model


def predict_ld(model, naca):
    features = np.array(parse_naca(naca)).reshape(1, -1)
    return model.predict(features)[0]


def predict_ld_with_uncertainty(model, naca):
    features = np.array(parse_naca(naca)).reshape(1, -1)
    predictions = [tree.predict(features)[0] for tree in model.estimators_]
    mean = np.mean(predictions)
    std = np.std(predictions)
    return mean, std
