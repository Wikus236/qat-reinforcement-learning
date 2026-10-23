"""Plot the recorded ESP32 pendulum angle in the simulation graph's style.

Run from Pendulum: ../.venv/bin/python plot_hardware_pendulum.py
"""

from Pendulum.paths import APP_ROOT

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Pendulum.plotting.position_plot import position_figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        default=APP_ROOT / "results/hardware" / "pendulum_position.csv")
    args = parser.parse_args()
    with args.csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error("CSV contains no samples.")
    times = np.array([float(row["time_s"]) for row in rows])
    angles = np.array([float(row["angle_deg"]) for row in rows])
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(angles)):
        parser.error("Time and angle must contain finite numbers.")
    if np.any(np.diff(times) <= 0):
        parser.error("Sample times must increase strictly.")
    fig = position_figure(times, angles, "Hardware pendulum position over time", wrap=True)
    for extension in ("png", "pdf"):
        path = args.csv.with_suffix("." + extension)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)
    print(f"Plotted {len(rows)} samples spanning {times[0]:.6f} to {times[-1]:.6f} seconds.")


if __name__ == "__main__":
    main()
