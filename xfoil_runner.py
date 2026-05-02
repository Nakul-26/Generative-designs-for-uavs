import re
import subprocess


def _extract_coefficient(pattern, output):
    match = re.search(pattern, output, re.IGNORECASE)
    if match is None:
        return None

    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def run_xfoil(airfoil="NACA 2412", reynolds=1000000, alpha=5, timeout=60):
    commands = f"""
{airfoil}
OPER
VISC {reynolds}
ALFA {alpha}
QUIT
"""

    try:
        completed = subprocess.run(
            ["xfoil.exe"],
            input=commands,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, None

    output = "\n".join(
        part for part in [completed.stdout or "", completed.stderr or ""] if part
    )

    cl = _extract_coefficient(r"CL\s*=\s*([-+0-9.eE]+)", output)
    cd = _extract_coefficient(r"CD\s*=\s*([-+0-9.eE]+)", output)

    return cl, cd
