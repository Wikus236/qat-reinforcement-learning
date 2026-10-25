from LunarLander.paths import APP_ROOT
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from LunarLander.deployment.esp_benchmark_data import (
    MODEL_LABELS,
    MODEL_ORDER,
    aggregate_results,
    load_rows,
)


DEFAULT_INPUT = Path(f"{APP_ROOT}/results/data/esp_benchmark_results.csv")
DEFAULT_OUTPUT_DIR = Path(f"{APP_ROOT}/results/figures/esp32")


plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 23,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
    "figure.titlesize": 26,
})


SERIES_STYLES = {
    "float32": {
        "marker": "o",
        "linestyle": "-",
    },
    "int8": {
        "marker": "s",
        "linestyle": "--",
    },
}


def plot_metric(
    rows,
    mean_key,
    std_key,
    ylabel,
    title,
    output_path,
    reference_line=None,
):
    figure, axis = plt.subplots(
        figsize=(11, 7),
    )
    all_sizes = sorted({row["size"] for row in rows})

    for model_type in MODEL_ORDER:
        model_rows = [
            row
            for row in rows
            if row["model_type"] == model_type
        ]

        if not model_rows:
            continue

        model_rows.sort(key=lambda row: row["size"])
        sizes = np.asarray(
            [row["size"] for row in model_rows],
            dtype=float,
        )
        means = np.asarray(
            [row[mean_key] for row in model_rows],
            dtype=float,
        )
        standard_deviations = np.asarray(
            [row[std_key] for row in model_rows],
            dtype=float,
        )

        axis.errorbar(
            sizes,
            means,
            yerr=standard_deviations,
            linewidth=2,
            markersize=8,
            capsize=5,
            label=MODEL_LABELS.get(model_type, model_type),
            **SERIES_STYLES.get(model_type, {}),
        )

    if reference_line is not None:
        axis.axhline(
            reference_line,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"Solved threshold ({reference_line:g})",
        )

    axis.set_xticks(all_sizes)
    axis.set_xlabel("Hidden layer size")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(
        True,
        alpha=0.3,
    )
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark results measured on the ESP32."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()

    rows = aggregate_results(
        load_rows(args.input)
    )
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_metric(
        rows,
        "mean_reward",
        "std_reward",
        "Mean reward",
        "ESP32 Mean Reward vs Model Size",
        args.output_dir / "esp_mean_reward.png",
        reference_line=200.0,
    )
    plot_metric(
        rows,
        "device_inference_mean_ms",
        "device_inference_std_ms",
        "On-device inference time (ms)",
        "ESP32 Inference Time vs Model Size",
        args.output_dir / "esp_device_inference_time.png",
    )
    plot_metric(
        rows,
        "serial_roundtrip_mean_ms",
        "serial_roundtrip_std_ms",
        "Serial round-trip time (ms)",
        "ESP32 Serial Round-Trip Time vs Model Size",
        args.output_dir / "esp_serial_roundtrip_time.png",
    )

    print(f"Saved ESP32 plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
