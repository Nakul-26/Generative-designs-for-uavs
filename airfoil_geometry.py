import numpy as np


def generate_naca4_coordinates(naca, n_points=100):
    digits = naca.replace("NACA ", "")

    m = int(digits[0]) / 100
    p = int(digits[1]) / 10
    t = int(digits[2:]) / 100

    x = np.linspace(0, 1, n_points)

    yt = 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    if p == 0:
        theta = np.zeros_like(x)
        xu = x
        yu = yt
        xl = x
        yl = -yt
        return xu, yu, xl, yl

    for idx in range(len(x)):
        if x[idx] < p:
            yc[idx] = m / (p**2) * (2 * p * x[idx] - x[idx] ** 2)
            dyc_dx[idx] = 2 * m / (p**2) * (p - x[idx])
        else:
            yc[idx] = m / ((1 - p) ** 2) * (
                (1 - 2 * p) + 2 * p * x[idx] - x[idx] ** 2
            )
            dyc_dx[idx] = 2 * m / ((1 - p) ** 2) * (p - x[idx])

    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)

    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl
