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
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 19,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 15,
    "figure.titlesize": 22,
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
        ax.grid(
            True,
            alpha=0.3,
        )

    flat_axes[-1].axis("off")

    figure.suptitle(
        "LunarLander DQN Additional Training Curves by Layer Size"
    )
    figure.supxlabel(
        "Additional Training Timesteps",
        y=0.065,
    )
    figure.supylabel(
        "Mean Evaluation Reward"
    )

    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.015),
    )

    figure.tight_layout(
        rect=(0.04, 0.11, 1, 0.95)
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
