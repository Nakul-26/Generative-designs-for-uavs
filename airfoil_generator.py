import random


def generate_random_naca():
    # First digit: max camber (0-9)
    camber = random.randint(0, 9)

    # Second digit: camber position (1-9)
    camber_position = random.randint(1, 9)

    # Last two digits: thickness (6-18%)
    thickness = random.randint(6, 18)

    return f"NACA {camber}{camber_position}{thickness:02d}"
