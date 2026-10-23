from LunarLander.paths import APP_ROOT
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Training curve folders
# ---------------------------------------------------------

CURVE_FOLDERS = {
    "Base": Path(f"{APP_ROOT}/models/initial_fp32/training_curves"),
    "FP32": Path(f"{APP_ROOT}/models/extended_fp32/training_curves"),
    "W8 QAT": Path(f"{APP_ROOT}/models/qat_w8/training_curves"),
    "W8A8 QAT": Path(f"{APP_ROOT}/models/qat_w8a8/training_curves"),
    "W4 QAT": Path(f"{APP_ROOT}/models/qat_w4/training_curves"),
    "W4A4 QAT": Path(f"{APP_ROOT}/models/qat_w4a4/training_curves"),
}

# ---------------------------------------------------------
# Load all CSV files
# ---------------------------------------------------------

def load_curves(folder):
    frames = []

    for csv_path in sorted(folder.glob("*.csv")):
        df = pd.read_csv(csv_path)

        frames.append(df)

    if not frames:
        return None

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ---------------------------------------------------------
# Plot one figure for each network size
# ---------------------------------------------------------

def plot_size(size):
    plt.figure(figsize=(10, 6))

    found_data = False

    for label, folder in CURVE_FOLDERS.items():
        df = load_curves(folder)

        if df is None:
            print(f"No curves found for {label}: {folder}")
            continue

        # Only use this network size
        df = df[
            df["size"] == size
        ].copy()

        if df.empty:
            print(f"No {size}x{size} curves found for {label}")
            continue

        found_data = True

        # -------------------------------------------------
        # Combine training seeds
        # -------------------------------------------------

        summary = (
            df.groupby("training_steps")["mean_reward"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("training_steps")
        )

        steps = summary["training_steps"].to_numpy()
        mean = summary["mean"].to_numpy()

        # std can be NaN if only one seed exists
        std = summary["std"].fillna(0).to_numpy()

        # -------------------------------------------------
        # Plot mean
        # -------------------------------------------------

        plt.plot(
            steps,
            mean,
            marker="o",
            linewidth=2,
            label=label,
        )

        # -------------------------------------------------
        # Plot ±1 std across training seeds
        # -------------------------------------------------

        plt.fill_between(
            steps,
            mean - std,
            mean + std,
            alpha=0.15,
        )

    if not found_data:
        plt.close()
        return

    plt.axhline(
        200,
        linestyle="--",
        linewidth=1,
        label="Solved threshold (200)",
    )

    plt.xlabel("Training Timesteps")
    plt.ylabel("Mean Evaluation Reward")

    plt.title(
        f"LunarLander DQN Training Curves — {size}x{size}"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    output_dir = Path(f"{APP_ROOT}/results/figures/training")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"training_curve_{size}_{size}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    print(f"Saved {output_path}")

    plt.show()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    # Change this if needed, or import MODEL_SIZES
    MODEL_SIZES = [
        16,
        32,
        64,
        128,
        256,
    ]

    for size in MODEL_SIZES:
        plot_size(size)
