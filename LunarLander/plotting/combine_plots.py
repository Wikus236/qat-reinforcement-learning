from LunarLander.paths import APP_ROOT
from pathlib import Path

import matplotlib.pyplot as plt

from LunarLander.plotting.plot_steps import (
    ADDITIONAL_TRAINING_LABELS,
    CURVE_FOLDERS,
    load_curves,
    plot_summary_on_axis,
    summarise_curve,
)


plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 23,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
    "figure.titlesize": 26,
})


MODEL_SIZES = [
    16,
    32,
    64,
    128,
    256,
]


def main():
    curves = {
        label: load_curves(CURVE_FOLDERS[label])
        for label in ADDITIONAL_TRAINING_LABELS
    }

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(18, 15),
        sharex=True,
        sharey=True,
    )
    flat_axes = axes.ravel()

    for ax, size in zip(flat_axes, MODEL_SIZES):
        for label in ADDITIONAL_TRAINING_LABELS:
            dataframe = curves[label]

            if dataframe is None:
                print(
                    f"No curves found for {label}: "
                    f"{CURVE_FOLDERS[label]}"
                )
                continue

            summary = summarise_curve(
                dataframe,
                size,
            )

            if summary is None:
                print(
                    f"No {size}x{size} curves found "
                    f"for {label}"
                )
                continue

            plot_summary_on_axis(
                ax,
                summary,
                label,
            )

        ax.axhline(
            200,
            color="black",
            linestyle="--",
            linewidth=1,
        )
        ax.set_title(f"{size}x{size}")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.grid(
            True,
            alpha=0.3,
        )

    for ax in flat_axes[:-1]:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(max(ymin, -1000), min(ymax, 300))

    flat_axes[-1].axis("off")

    figure.suptitle(
        "LunarLander DQN additional training\nby hidden-layer width"
    )
    figure.supxlabel(
        "Additional Training Timesteps",
        y=0.035,
    )
    figure.supylabel(
        "Mean Evaluation Reward"
    )

    for ax in flat_axes[:-1]:
        ax.legend(loc="best", ncol=3, columnspacing=0.7,
                  handlelength=1.4, handletextpad=0.4, labelspacing=0.3)

    figure.subplots_adjust(
        left=0.16, right=0.98, bottom=0.11, top=0.83,
        hspace=0.5, wspace=0.12,
    )

    output_dir = Path(f"{APP_ROOT}/results/figures/training")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path = (
        output_dir
        / "training_curve_additional_all_sizes.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
    )
    plt.close(figure)

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
