import shutil
import subprocess
from pathlib import Path


def main():
    dashboard_path = Path(__file__).with_name("dashboard.py")
    streamlit_executable = shutil.which("streamlit")

    if streamlit_executable is None:
        raise SystemExit(
            "The Matplotlib visualizer has been replaced.\n"
            "Install Streamlit with `pip install streamlit`, then run:\n"
            f"streamlit run {dashboard_path.name}"
        )

    subprocess.run([streamlit_executable, "run", str(dashboard_path)], check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
