import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor


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

    for airfoil, ld in data.items():
        features = parse_naca(airfoil)
        X.append(features)
        y.append(ld)

    return np.array(X), np.array(y)


def train_model():
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
