import json
import time
from pathlib import Path

from genetic_optimizer import run_ga


CONTROL_FILE = Path("control.json")


def is_running():
    if not CONTROL_FILE.exists():
        return True

    try:
        with open(CONTROL_FILE, "r") as handle:
            return bool(json.load(handle).get("running", True))
    except (OSError, json.JSONDecodeError):
        return True


if __name__ == "__main__":
    cycle = 1
    waiting_for_start = False

    while True:
        if not is_running():
            if not waiting_for_start:
                print("Stopped. Waiting to start...")
                waiting_for_start = True
            time.sleep(1)
            continue

        waiting_for_start = False
        print(f"\n=== Optimization Cycle {cycle} ===\n")
        run_ga()
        cycle += 1
        time.sleep(2)
