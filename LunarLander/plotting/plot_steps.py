"""Generate the initial, additional and full-history LunarLander training plots."""

from pathlib import Path
import os
import textwrap

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import pandas as pd

from LunarLander.paths import APP_ROOT
from LunarLander.training.train_base_initial import MODEL_SIZES

CURVE_FOLDERS = {
    "Base": APP_ROOT / "models/initial_fp32/training_curves",
    "FP32": APP_ROOT / "models/extended_fp32/training_curves",
    "W8 QAT": APP_ROOT / "models/qat_w8/training_curves",
    "W8A8 QAT": APP_ROOT / "models/qat_w8a8/training_curves",
    "W4 QAT": APP_ROOT / "models/qat_w4/training_curves",
    "W4A4 QAT": APP_ROOT / "models/qat_w4a4/training_curves",
}
ADDITIONAL_TRAINING_LABELS = tuple(label for label in CURVE_FOLDERS if label != "Base")
OUTPUT_DIR = APP_ROOT / "results/figures/training"
plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 24, "axes.labelsize": 23,
    "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 18,
    "figure.titlesize": 26,
})


def load_curves(folder):
    frames = [pd.read_csv(path) for path in sorted(Path(folder).glob("*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else None


def summarise_curve(dataframe, size):
    if dataframe is None:
        return None
    selected = dataframe[dataframe["size"] == size]
    if selected.empty:
        return None
    summary = selected.groupby("training_steps")["mean_reward"].agg(
        ["mean", "std", "count"]
    ).reset_index().sort_values("training_steps")
    summary["std"] = summary["std"].fillna(0)
    return summary


def plot_summary_on_axis(ax, summary, label, step_offset=0, **kwargs):
    steps = summary["training_steps"].to_numpy(dtype=float) + step_offset
    mean = summary["mean"].to_numpy(dtype=float)
    std = summary["std"].to_numpy(dtype=float)
    line, = ax.plot(steps, mean, linewidth=2, label=label, **kwargs)
    ax.fill_between(steps, mean - std, mean + std, color=line.get_color(), alpha=0.15)


def finish_plot(fig, ax, title, xlabel, path):
    ax.axhline(200, color="black", linestyle="--", linewidth=1,
               label="Reward-200 reference")
    ax.set(title=textwrap.fill(title, 45), xlabel=textwrap.fill(xlabel, 46),
           ylabel="Mean evaluation reward")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(max(ymin, -1000), min(ymax, 300))
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.legend(loc="best", ncol=2, columnspacing=1.0, handletextpad=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_size(size, curves=None, output_dir=OUTPUT_DIR):
    curves = curves or {label: load_curves(folder) for label, folder in CURVE_FOLDERS.items()}
    base = summarise_curve(curves["Base"], size)
    if base is None:
        raise ValueError(f"Missing initial FP32 training curve for {size}x{size}")
    offset = int(base["training_steps"].max())
    fig, ax = plt.subplots(figsize=(12, 9))
    for label in ADDITIONAL_TRAINING_LABELS:
        summary = summarise_curve(curves[label], size)
        if summary is None:
            raise ValueError(f"Missing {label} training curve for {size}x{size}")
        plot_summary_on_axis(ax, summary, label)
    finish_plot(fig, ax, f"LunarLander {size}x{size}: additional training",
                "Additional training steps", output_dir / f"training_curve_additional_{size}_{size}.png")

    fig, ax = plt.subplots(figsize=(12, 9))
    plot_summary_on_axis(ax, base, "Initial FP32", color="black")
    for label in ADDITIONAL_TRAINING_LABELS:
        summary = summarise_curve(curves[label], size)
        plot_summary_on_axis(ax, summary, label, step_offset=offset)
    ax.axvline(offset, color="grey", linestyle=":", linewidth=1)
    finish_plot(fig, ax, f"LunarLander {size}x{size}: initial and additional training",
                "Training steps (additional stage offset by initial budget)",
                output_dir / f"training_curve_full_{size}_{size}.png")


def main():
    curves = {label: load_curves(folder) for label, folder in CURVE_FOLDERS.items()}
    group_plots = {
        "Base": ("initial", "Initial FP32"),
        "FP32": ("fp32", "FP32 additional"),
        "W8 QAT": ("w8_qat", "W8 QAT additional"),
        "W8A8 QAT": ("w8a8_qat", "W8A8 QAT additional"),
        "W4 QAT": ("w4_qat", "W4 QAT additional"),
        "W4A4 QAT": ("w4a4_qat", "W4A4 QAT additional"),
    }
    for label, (filename, title) in group_plots.items():
        fig, ax = plt.subplots(figsize=(12, 9))
        for size in MODEL_SIZES:
            summary = summarise_curve(curves[label], size)
            if summary is None:
                raise ValueError(f"Missing {label} training curve for {size}x{size}")
            plot_summary_on_axis(ax, summary, f"{size}x{size}")
        finish_plot(fig, ax, f"LunarLander DQN {title} training curves",
                    "Training steps" if label == "Base" else "Additional training steps",
                    OUTPUT_DIR / f"training_curve_{filename}_all_sizes.png")
    for size in MODEL_SIZES:
        plot_size(size, curves=curves)


if __name__ == "__main__":
    main()
