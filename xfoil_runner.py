import subprocess
import re

def run_xfoil(airfoil="NACA 2412", reynolds=1000000, alpha=5):

    commands = f"""
{airfoil}
OPER
VISC {reynolds}
ALFA {alpha}
QUIT
"""

    process = subprocess.Popen(
        "xfoil.exe",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    output, error = process.communicate(commands)

    # Extract CL and CD
    cl_match = re.search(r"CL\s*=\s*([-0-9.]+)", output)
    cd_match = re.search(r"CD\s*=\s*([-0-9.]+)", output)

    CL = float(cl_match.group(1)) if cl_match else None
    CD = float(cd_match.group(1)) if cd_match else None

    return CL, CD