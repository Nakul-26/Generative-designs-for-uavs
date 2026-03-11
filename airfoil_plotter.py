from pathlib import Path

import matplotlib.pyplot as plt

from airfoil_geometry import generate_naca4_coordinates


def plot_airfoil(naca):
    xu, yu, xl, yl = generate_naca4_coordinates(naca)

    plt.figure(figsize=(8, 3))
    plt.plot(xu, yu)
    plt.plot(xl, yl)
    plt.title(f"Optimized Airfoil: {naca}")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(Path("best_airfoil.png"), dpi=150, bbox_inches="tight")
    plt.close()
