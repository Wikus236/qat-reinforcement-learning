from LunarLander.paths import APP_ROOT
import csv
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 23,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
    "figure.titlesize": 26,
})


CSV_PATH = Path(
    f"{APP_ROOT}/results/data/model_benchmark_results.csv"
)

OUTPUT_DIR = Path(
    f"{APP_ROOT}/results/figures/benchmark"
)


SERIES_ORDER = [
    ("base", "FP32 custom"),
    ("weight_int8", "W8 exported"),
    (
        "weight_int4",
        "W4 simulated, INT8 storage",
    ),
    (
        "weight_activation_int8",
        "W8A8 exported",
    ),
    (
        "weight_activation_int4",
        "W4A4 simulated, INT8 storage",
    ),
]

SPLIT_SERIES_PERFORMANCE = [
    (
        "INT8 exports",
        [
            ("base", "FP32 custom"),
            ("weight_int8", "W8 exported"),
            ("weight_activation_int8", "W8A8 exported"),
        ],
    ),
    (
        "4-bit simulated",
        [
            ("base", "FP32 custom"),
            ("weight_int4", "W4 simulated"),
            ("weight_activation_int4", "W4A4 simulated"),
        ],
    ),
]

SPLIT_SERIES_MEASURED = [
    (
        "INT8 exports",
        [
            ("base", "FP32 custom"),
            ("weight_int8", "W8 exported"),
            ("weight_activation_int8", "W8A8 exported"),
        ],
    ),
    (
        "4-bit simulated, stored as INT8",
        [
            ("base", "FP32 custom"),
            ("weight_int4", "W4 simulated, INT8 storage"),
            ("weight_activation_int4", "W4A4 simulated, INT8 storage"),
        ],
    ),
]

SPLIT_SERIES_THEORETICAL = [
    (
        "INT8 exported storage",
        [
            ("base", "FP32 custom"),
            ("weight_int8", "W8 exported"),
            ("weight_activation_int8", "W8A8 exported"),
        ],
    ),
    (
        "Theoretical packed INT4 storage",
        [
            ("base", "FP32 custom"),
            ("weight_int4", "W4 theoretical packed INT4"),
            ("weight_activation_int4", "W4A4 theoretical packed INT4"),
        ],
    ),
]

SPLIT_COMPARISON_PERFORMANCE = [
    (
        "INT8 exports",
        [
            ("weight_int8", "W8 exported"),
            ("weight_activation_int8", "W8A8 exported"),
        ],
    ),
    (
        "4-bit simulated",
        [
            ("weight_int4", "W4 simulated"),
            ("weight_activation_int4", "W4A4 simulated"),
        ],
    ),
]

SPLIT_COMPARISON_MEASURED = [
    (
        "INT8 exports",
        [
            ("weight_int8", "W8 exported"),
            ("weight_activation_int8", "W8A8 exported"),
        ],
    ),
    (
        "4-bit simulated, stored as INT8",
        [
            ("weight_int4", "W4 simulated, INT8 storage"),
            ("weight_activation_int4", "W4A4 simulated, INT8 storage"),
        ],
    ),
]

SPLIT_COMPARISON_THEORETICAL = [
    (
        "INT8 exported storage",
        [
            ("weight_int8", "W8 exported"),
            ("weight_activation_int8", "W8A8 exported"),
        ],
    ),
    (
        "Theoretical packed INT4 storage",
        [
            ("weight_int4", "W4 theoretical packed INT4"),
            ("weight_activation_int4", "W4A4 theoretical packed INT4"),
        ],
    ),
]


SERIES_STYLES = {
    "base": {
        "marker": "o",
        "linestyle": "-",
    },
    "weight_int8": {
        "marker": "s",
        "linestyle": "--",
    },
    "weight_int4": {
        "marker": "^",
        "linestyle": ":",
    },
    "weight_activation_int8": {
        "marker": "D",
        "linestyle": "-.",
    },
    "weight_activation_int4": {
        "marker": "P",
        "linestyle": (0, (3, 1, 1, 1)),
    },
}


def offset_x(x, key, groups):
    x = np.array(
        x,
        dtype=float,
    )

    if len(x) == 0:
        return x

    keys = [
        group_key
        for group_key, _ in groups
    ]

    if key not in keys:
        return x

    unique_x = sorted(
        set(x)
    )

    if len(unique_x) > 1:
        min_gap = min(
            np.diff(unique_x)
        )
    else:
        min_gap = max(
            abs(unique_x[0]) * 0.05,
            1.0,
        )

    spread = min_gap * 0.24
    step = spread / max(
        len(keys) - 1,
        1,
    )
    index = keys.index(key)
    center = (
        len(keys) - 1
    ) / 2

    return x + (
        index - center
    ) * step


def plot_series(
    ax,
    x,
    y,
    key,
    label,
    groups,
    yerr=None,
    separate_points=True,
):
    style = SERIES_STYLES.get(
        key,
        {},
    )

    if separate_points:
        plot_x = offset_x(
            x,
            key,
            groups,
        )
    else:
        plot_x = np.array(
            x,
            dtype=float,
        )

    line, = ax.plot(
        plot_x,
        y,
        linewidth=2,
        markersize=6,
        markeredgewidth=1.2,
        markeredgecolor="black",
        markerfacecolor="white",
        alpha=0.95,
        label=label,
        **style,
    )

    if yerr is not None:
        ax.fill_between(
            plot_x,
            y - yerr,
            y + yerr,
            color=line.get_color(),
            alpha=0.12,
        )

    return line


def make_split_figure(title):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 8),
        sharey=True,
    )

    figure.suptitle(textwrap.fill(title, width=70))

    return figure, axes


def finish_split_axis(
    ax,
    sizes,
    ylabel,
    panel_title,
    log_y=False,
):
    if log_y:
        ax.set_yscale("log")

    ax.set_xticks(sizes)
    ax.set_xlabel(
        "Hidden layer size"
    )
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title)
    ax.grid(
        True,
        alpha=0.3,
    )
    ax.legend()


def save_split_figure(figure, filename):
    for axis in figure.axes:
        axis.set_ylabel(textwrap.fill(axis.get_ylabel(), width=35))
        axis.set_xlabel(textwrap.fill(axis.get_xlabel(), width=35))
        axis.set_title(textwrap.fill(axis.get_title(), width=35))
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / filename,
        dpi=250,
    )
    plt.close(figure)


def normalize_group(model_type):
    mapping = {
        "base_float":
            "base",

        "qat_weight_only":
            "weight_qat",

        "int8_weight_only":
            "weight_int8",

        "qat_weight_only_int4":
            "weight_qat_int4",

        "int4_weight_only":
            "weight_int4",

        "qat_weight_activation":
            "weight_activation_qat",

        "int8_weight_activation":
            "weight_activation_int8",

        "qat_weight_activation_int4":
            "weight_activation_qat_int4",

        "int4_weight_activation":
            "weight_activation_int4",
    }

    return mapping.get(
        model_type,
        model_type,
    )


def theoretical_activation_memory_mb(
    size,
    group,
    batch_size=1,
):
    hidden_activation_elements = size * 2

    if group in (
        "weight_activation_int8",
        "weight_activation_qat",
    ):
        bytes_per_activation = 1.0
    elif group in (
        "weight_activation_int4",
        "weight_activation_qat_int4",
    ):
        bytes_per_activation = 0.5
    else:
        bytes_per_activation = 4.0

    total_bytes = (
        hidden_activation_elements
        * batch_size
        * bytes_per_activation
    )

    return total_bytes / (1024 ** 2)


def add_derived_runtime_memory_columns(row):
    size = int(row["size"])
    group = normalize_group(
        row["model_type"]
    )

    if not row.get(
        "theoretical_activation_memory_mb"
    ):
        row["theoretical_activation_memory_mb"] = (
            theoretical_activation_memory_mb(
                size,
                group,
            )
        )

    if not row.get(
        "theoretical_runtime_memory_mb"
    ):
        row["theoretical_runtime_memory_mb"] = (
            float(row["theoretical_parameter_memory_mb"])
            + float(row["theoretical_activation_memory_mb"])
        )

    return row


def load_rows():
    with CSV_PATH.open(
        newline=""
    ) as handle:

        return [
            add_derived_runtime_memory_columns(row)
            for row in csv.DictReader(handle)
        ]


def group_rows(rows):
    grouped = defaultdict(
        lambda: defaultdict(list)
    )

    for row in rows:
        size = int(row["size"])

        group = normalize_group(
            row["model_type"]
        )

        grouped[size][group].append(
            row
        )

    return grouped


def row_value(row, metric):
    # Source benchmark CSVs remain in milliseconds; convert only for display.
    time_sources = {
        "inference_mean_us": "inference_mean_ms",
        "inference_std_us": "inference_std_ms",
    }
    if metric in time_sources:
        return 1000.0 * row_value(row, time_sources[metric])

    if metric in row and row[metric] != "":
        return float(row[metric])

    old_column_names = {
        "solve_rate": "success_rate",
    }

    old_metric = old_column_names.get(metric)
    if old_metric and old_metric in row and row[old_metric] != "":
        return float(row[old_metric])

    raise KeyError(metric)


def values(rows, metric):
    return np.array(
        [
            row_value(row, metric)
            for row in rows
        ],
        dtype=float,
    )


def plot_detailed_metric(
    grouped,
    metric,
    ylabel,
    filename,
    split_series,
    log_y=False,
):
    sizes = sorted(
        grouped.keys()
    )

    figure, axes = make_split_figure(
        f"{ylabel} - all trained models"
    )

    for ax, (panel_title, groups) in zip(
        axes,
        split_series,
    ):
        for key, label in groups:

            mean_x = []
            mean_y = []
            std_y = []

            for size in sizes:

                rows = grouped[size].get(
                    key,
                    [],
                )

                if not rows:
                    continue

                vals = values(
                    rows,
                    metric,
                )

                mean_x.append(size)

                mean_y.append(
                    np.mean(vals)
                )

                std_y.append(
                    np.std(vals)
                )

            if mean_x:
                mean_y = np.array(mean_y)
                std_y = np.array(std_y)

                plot_series(
                    ax,
                    mean_x,
                    mean_y,
                    key,
                    label,
                    groups,
                    yerr=std_y,
                )

        finish_split_axis(
            ax,
            sizes,
            ylabel,
            panel_title,
            log_y=log_y,
        )

    save_split_figure(
        figure,
        filename,
    )


def plot_summary_metric(
    grouped,
    metric,
    ylabel,
    filename,
    split_series,
    log_y=False,
):
    sizes = sorted(
        grouped.keys()
    )

    figure, axes = make_split_figure(
        f"{ylabel} - mean +/- SD across training seeds"
    )

    for ax, (panel_title, groups) in zip(
        axes,
        split_series,
    ):
        for key, label in groups:

            x = []
            y = []
            error = []

            for size in sizes:

                rows = grouped[size].get(
                    key,
                    [],
                )

                if not rows:
                    continue

                vals = values(
                    rows,
                    metric,
                )

                x.append(size)

                y.append(
                    np.mean(vals)
                )

                error.append(
                    np.std(vals)
                )

            if x:
                y = np.array(y)
                error = np.array(error)

                plot_series(
                    ax,
                    x,
                    y,
                    key,
                    label,
                    groups,
                    yerr=error,
                )

        finish_split_axis(
            ax,
            sizes,
            ylabel,
            panel_title,
            log_y=log_y,
        )

    if metric in (
        "inference_mean_us",
        "theoretical_activation_memory_mb",
        "theoretical_runtime_memory_mb",
    ):
        axes[1].set_ylabel("")

    save_split_figure(
        figure,
        filename,
    )


def plot_reduction_vs_fp32(
    grouped,
    metric,
    ylabel,
    title,
    filename,
    split_comparison_series,
):
    sizes = sorted(
        grouped.keys()
    )

    figure, axes = make_split_figure(title)

    for ax, (panel_title, comparison_groups) in zip(
        axes,
        split_comparison_series,
    ):
        for key, label in comparison_groups:

            x = []
            y = []

            for size in sizes:

                base_rows = grouped[size].get(
                    "base",
                    [],
                )

                quant_rows = grouped[size].get(
                    key,
                    [],
                )

                if (
                    not base_rows
                    or not quant_rows
                ):
                    continue

                base_value = np.mean(
                    values(
                        base_rows,
                        metric,
                    )
                )

                quant_value = np.mean(
                    values(
                        quant_rows,
                        metric,
                    )
                )

                if base_value == 0:
                    continue

                reduction = (
                    (
                        base_value
                        - quant_value
                    )
                    / base_value
                    * 100.0
                )

                x.append(size)
                y.append(reduction)

            if x:

                plot_series(
                    ax,
                    x,
                    y,
                    key,
                    label,
                    comparison_groups,
                )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1,
        )
        finish_split_axis(
            ax,
            sizes,
            ylabel,
            panel_title,
        )

    save_split_figure(
        figure,
        filename,
    )



def plot_reduction_bars_vs_fp32(
    grouped,
    metric,
    ylabel,
    title,
    filename,
    split_comparison_series,
):
    sizes = sorted(
        grouped.keys()
    )

    figure, axes = make_split_figure(title)

    for ax, (panel_title, comparison_groups) in zip(
        axes,
        split_comparison_series,
    ):
        x_positions = np.arange(
            len(sizes),
            dtype=float,
        )

        num_groups = len(comparison_groups)
        bar_width = 0.72 / max(
            num_groups,
            1,
        )

        for group_index, (key, label) in enumerate(comparison_groups):
            y = []

            for size in sizes:
                base_rows = grouped[size].get(
                    "base",
                    [],
                )

                quant_rows = grouped[size].get(
                    key,
                    [],
                )

                if (
                    not base_rows
                    or not quant_rows
                ):
                    y.append(np.nan)
                    continue

                base_value = np.mean(
                    values(
                        base_rows,
                        metric,
                    )
                )

                quant_value = np.mean(
                    values(
                        quant_rows,
                        metric,
                    )
                )

                if base_value == 0:
                    y.append(np.nan)
                    continue

                reduction = (
                    (base_value - quant_value)
                    / base_value
                    * 100.0
                )

                y.append(reduction)

            offset = (
                group_index
                - (num_groups - 1) / 2
            ) * bar_width

            style = SERIES_STYLES.get(
                key,
                {},
            )

            ax.bar(
                x_positions + offset,
                y,
                width=bar_width,
                label=label,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.8,
                hatch="//" if style.get("linestyle") == ":" else None,
            )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1,
            color="black",
            alpha=0.5,
        )
        ax.set_xticks(
            x_positions,
            [str(size) for size in sizes],
        )
        ax.set_xlabel(
            "Hidden layer size"
        )
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.grid(
            True,
            axis="y",
            alpha=0.3,
        )
        ax.legend(loc="lower right")

    save_split_figure(
        figure,
        filename,
    )


def plot_reward_retention(
    grouped,
):
    sizes = sorted(
        grouped.keys()
    )

    figure, axes = make_split_figure(
        "Reward retention relative to FP32"
    )

    for ax, (panel_title, comparison_groups) in zip(
        axes,
        SPLIT_COMPARISON_PERFORMANCE,
    ):
        for key, label in comparison_groups:

            x = []
            y = []

            for size in sizes:

                base_rows = grouped[size].get(
                    "base",
                    [],
                )

                other_rows = grouped[size].get(
                    key,
                    [],
                )

                if (
                    not base_rows
                    or not other_rows
                ):
                    continue

                base_reward = np.mean(
                    values(
                        base_rows,
                        "mean_reward",
                    )
                )

                other_reward = np.mean(
                    values(
                        other_rows,
                        "mean_reward",
                    )
                )

                if base_reward == 0:
                    continue

                retention = (
                    other_reward
                    / base_reward
                    * 100.0
                )

                x.append(size)
                y.append(retention)

            if x:

                plot_series(
                    ax,
                    x,
                    y,
                    key,
                    label,
                    comparison_groups,
                )

        ax.axhline(
            100,
            linestyle="--",
            linewidth=1,
        )
        finish_split_axis(
            ax,
            sizes,
            "Mean reward retention (%)",
            panel_title,
        )

    save_split_figure(
        figure,
        "reward_retention_vs_fp32.png",
    )


def plot_reward_vs_memory_metric(
    grouped,
    metric,
    xlabel,
    title,
    filename,
    split_series,
):
    figure, axes = make_split_figure(title)

    for ax, (panel_title, groups) in zip(
        axes,
        split_series,
    ):
        for key, label in groups:

            x = []
            y = []

            for size in sorted(
                grouped.keys()
            ):

                rows = grouped[size].get(
                    key,
                    [],
                )

                if not rows:
                    continue

                memory = np.mean(
                    values(
                        rows,
                        metric,
                    )
                )

                reward = np.mean(
                    values(
                        rows,
                        "mean_reward",
                    )
                )

                x.append(memory)
                y.append(reward)

            if x:

                plot_series(
                    ax,
                    x,
                    y,
                    key,
                    label,
                    groups,
                    separate_points=False,
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(
            "Mean reward"
        )
        ax.set_title(panel_title)
        ax.grid(
            True,
            alpha=0.3,
        )
        ax.legend()

    save_split_figure(
        figure,
        filename,
    )


def plot_reward_vs_memory(
    grouped,
):
    plot_reward_vs_memory_metric(
        grouped,
        "theoretical_parameter_memory_mb",
        "Theoretical packed parameter memory (MB)",
        "Reward vs theoretical packed parameter memory",
        "reward_vs_theoretical_memory.png",
        SPLIT_SERIES_THEORETICAL,
    )


def plot_reward_vs_runtime_memory(
    grouped,
):
    plot_reward_vs_memory_metric(
        grouped,
        "theoretical_runtime_memory_mb",
        "Theoretical parameter + activation memory (MB)",
        "Reward vs theoretical runtime memory",
        "reward_vs_theoretical_runtime_memory.png",
        SPLIT_SERIES_THEORETICAL,
    )


def plot_reward_vs_inference(
    grouped,
):
    figure, axes = make_split_figure(
        "Reward vs deployment-style NumPy inference time"
    )

    for ax, (panel_title, groups) in zip(
        axes,
        SPLIT_SERIES_MEASURED,
    ):
        for key, label in groups:

            x = []
            y = []

            for size in sorted(
                grouped.keys()
            ):

                rows = grouped[size].get(
                    key,
                    [],
                )

                if not rows:
                    continue

                inference = np.mean(
                    values(
                        rows,
                        "inference_mean_us",
                    )
                )

                reward = np.mean(
                    values(
                        rows,
                        "mean_reward",
                    )
                )

                x.append(inference)
                y.append(reward)

            if x:

                plot_series(
                    ax,
                    x,
                    y,
                    key,
                    label,
                    groups,
                    separate_points=False,
                )

        ax.set_xlabel(
            "Deployment-style NumPy inference time (µs)"
        )
        ax.set_ylabel(
            "Mean reward"
        )
        ax.set_title(panel_title)
        ax.grid(
            True,
            alpha=0.3,
        )
        ax.legend()

    save_split_figure(
        figure,
        "reward_vs_inference_time.png",
    )


def main():

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    rows = load_rows()

    grouped = group_rows(
        rows
    )

    # ==========================================
    # REWARD
    # ==========================================

    plot_detailed_metric(
        grouped,
        "mean_reward",
        "Mean reward",
        "detailed_mean_reward.png",
        SPLIT_SERIES_PERFORMANCE,
    )

    plot_summary_metric(
        grouped,
        "mean_reward",
        "Mean reward",
        "summary_mean_reward.png",
        SPLIT_SERIES_PERFORMANCE,
    )

    plot_summary_metric(
        grouped,
        "solve_rate",
        "Solve rate",
        "summary_solve_rate.png",
        SPLIT_SERIES_PERFORMANCE,
    )

    plot_summary_metric(
        grouped,
        "success_rate",
        "Successful landing rate",
        "summary_success_rate.png",
        SPLIT_SERIES_PERFORMANCE,
    )

    plot_summary_metric(
        grouped,
        "failure_rate",
        "Failure rate",
        "summary_failure_rate.png",
        SPLIT_SERIES_PERFORMANCE,
    )

    # ==========================================
    # FILE SIZE
    # ==========================================

    plot_summary_metric(
        grouped,
        "file_size_mb",
        "File size (MB)",
        "summary_file_size.png",
        SPLIT_SERIES_MEASURED,
    )

    plot_reduction_vs_fp32(
        grouped,
        "file_size_mb",
        "File size reduction vs FP32 (%)",
        "File size reduction relative to FP32",
        "file_size_reduction_percent.png",
        SPLIT_COMPARISON_MEASURED,
    )

    # ==========================================
    # ACTUAL STORAGE USED BY CURRENT PYTHON
    # REPRESENTATION
    # ==========================================

    plot_summary_metric(
        grouped,
        "parameter_memory_mb",
        "Actual stored parameter memory (MB)",
        "summary_stored_parameter_memory.png",
        SPLIT_SERIES_MEASURED,
    )

    # ==========================================
    # THEORETICAL DEPLOYMENT MEMORY
    #
    # This is the important memory result:
    #
    # FP32 = 32 bits / weight
    # INT8 = 8 bits / weight
    # INT4 storage here is theoretical packed storage.
    # The actual exported INT4 tensors are still stored as INT8.
    # ==========================================

    plot_detailed_metric(
        grouped,
        "theoretical_parameter_memory_mb",
        "Theoretical packed deployment memory (MB)",
        "detailed_theoretical_memory.png",
        SPLIT_SERIES_THEORETICAL,
    )

    plot_summary_metric(
        grouped,
        "theoretical_parameter_memory_mb",
        "Theoretical packed deployment memory (MB)",
        "summary_theoretical_memory.png",
        SPLIT_SERIES_THEORETICAL,
    )

    plot_reduction_vs_fp32(
        grouped,
        "theoretical_parameter_memory_mb",
        "Memory reduction vs FP32 (%)",
        (
            "Theoretical packed deployment memory "
            "reduction relative to FP32"
        ),
        "theoretical_memory_reduction_percent.png",
        SPLIT_COMPARISON_THEORETICAL,
    )

    plot_reduction_bars_vs_fp32(
        grouped,
        "theoretical_parameter_memory_mb",
        "Memory reduction vs FP32 (%)",
        (
            "Theoretical packed deployment memory "
            "reduction relative to FP32"
        ),
        "theoretical_memory_reduction_bar_percent.png",
        SPLIT_COMPARISON_THEORETICAL,
    )

    # ==========================================
    # THEORETICAL RUNTIME ACTIVATION MEMORY
    # ==========================================

    plot_summary_metric(
        grouped,
        "theoretical_activation_memory_mb",
        "Theoretical activation memory per inference (MB)",
        "summary_theoretical_activation_memory.png",
        SPLIT_SERIES_THEORETICAL,
    )

    plot_reduction_vs_fp32(
        grouped,
        "theoretical_activation_memory_mb",
        "Activation memory reduction vs FP32 (%)",
        "Theoretical activation memory reduction relative to FP32",
        "theoretical_activation_memory_reduction_percent.png",
        SPLIT_COMPARISON_THEORETICAL,
    )

    plot_summary_metric(
        grouped,
        "theoretical_runtime_memory_mb",
        "Theoretical parameter + activation memory (MB)",
        "summary_theoretical_runtime_memory.png",
        SPLIT_SERIES_THEORETICAL,
    )

    plot_reduction_vs_fp32(
        grouped,
        "theoretical_runtime_memory_mb",
        "Runtime memory reduction vs FP32 (%)",
        "Theoretical runtime memory reduction relative to FP32",
        "theoretical_runtime_memory_reduction_percent.png",
        SPLIT_COMPARISON_THEORETICAL,
    )

    # ==========================================
    # INFERENCE TIME
    # ==========================================

    plot_detailed_metric(
        grouped,
        "inference_mean_us",
        "Deployment-style NumPy inference time (µs)",
        "detailed_inference_time.png",
        SPLIT_SERIES_MEASURED,
    )

    plot_summary_metric(
        grouped,
        "inference_mean_us",
        "Deployment-style NumPy inference time (µs)",
        "summary_inference_time.png",
        SPLIT_SERIES_MEASURED,
    )

    plot_summary_metric(
        grouped,
        "inference_std_us",
        "Deployment-style NumPy inference timing SD (µs)",
        "summary_inference_timing_sd.png",
        SPLIT_SERIES_MEASURED,
    )

    plot_reduction_vs_fp32(
        grouped,
        "inference_mean_ms",
        "Inference time reduction vs FP32 (%)",
        "Deployment-style NumPy inference time reduction relative to FP32",
        "inference_time_reduction_percent.png",
        SPLIT_COMPARISON_MEASURED,
    )

    # ==========================================
    # PYTHON MEMORY
    #
    # Do not interpret this as deployment RAM.
    # tracemalloc only measures Python-managed
    # allocations.
    # ==========================================

    plot_summary_metric(
        grouped,
        "peak_python_memory_mb",
        "Peak Python memory (MB)",
        "summary_peak_python_memory.png",
        SPLIT_SERIES_MEASURED,
    )

    # ==========================================
    # TRADE-OFF PLOTS
    # ==========================================

    plot_reward_retention(
        grouped
    )

    plot_reward_vs_memory(
        grouped
    )

    plot_reward_vs_runtime_memory(
        grouped
    )

    plot_reward_vs_inference(
        grouped
    )

    print(
        f"Saved benchmark plots to "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
