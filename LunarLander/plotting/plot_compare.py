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


DETAILED_CSV_PATH = Path(f"{APP_ROOT}/results/data/model_results_detailed.csv")
SUMMARY_CSV_PATH = Path(f"{APP_ROOT}/results/data/model_results_summary.csv")
OUTPUT_DIR = Path(f"{APP_ROOT}/results/figures/comparison")


SERIES_ORDER = [
    ("base", "FP32 custom"),
    ("weight_qat", "W8 QAT"),
    ("weight_int8", "W8 exported"),
    ("weight_qat_int4", "W4 QAT"),
    ("weight_int4", "W4 exported"),
    ("weight_activation_qat", "W8A8 QAT"),
    ("weight_activation_int8", "W8A8 exported"),
    ("weight_activation_qat_int4", "W4A4 QAT"),
    ("weight_activation_int4", "W4A4 exported"),
]


def normalize_group(model_type: str) -> str:
    mapping = {
        "base_float": "base",
        "qat_weight_only": "weight_qat",
        "int8_weight_only": "weight_int8",
        "qat_weight_only_int4": "weight_qat_int4",
        "int4_weight_only": "weight_int4",
        "qat_weight_activation": "weight_activation_qat",
        "int8_weight_activation": "weight_activation_int8",
        "qat_weight_activation_int4": "weight_activation_qat_int4",
        "int4_weight_activation": "weight_activation_int4",
    }

    return mapping.get(model_type, model_type)


def load_rows(csv_path: Path):
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def has_column(rows, key: str) -> bool:
    return any(
        row.get(key, "") != ""
        for row in rows
    )


def to_float(row, key: str) -> float:
    if key in row and row[key] != "":
        return float(row[key])

    old_column_names = {
        "solve_rate": "success_rate",
        "mean_solve_rate": "mean_success_rate",
    }

    old_key = old_column_names.get(key)
    if old_key and old_key in row and row[old_key] != "":
        return float(row[old_key])

    raise KeyError(key)


def group_detailed_rows(rows):
    grouped = defaultdict(lambda: defaultdict(list))

    for row in rows:
        size = int(row["size"])
        group = normalize_group(row["model_type"])

        grouped[size][group].append(row)

    return grouped


def group_summary_rows(rows):
    grouped = defaultdict(dict)

    for row in rows:
        size = int(row["size"])
        group = normalize_group(row["model_type"])

        grouped[size][group] = row

    return grouped


def plot_detailed_metric(
    grouped,
    metric: str,
    ylabel: str,
    filename: str,
):
    sizes = sorted(grouped.keys())

    plt.figure(figsize=(11, 7))

    for key, label in SERIES_ORDER:
        if metric == "mean_steps" and "qat" in key:
            continue

        mean_x = []
        mean_y = []
        std_y = []

        for size in sizes:
            rows = grouped[size].get(key, [])

            if not rows:
                continue

            vals = [
                to_float(row, metric)
                for row in rows
            ]

            mean_x.append(size)
            mean_y.append(np.mean(vals))
            std_y.append(np.std(vals))

        if mean_x:
            mean_y = np.array(mean_y)
            std_y = np.array(std_y)

            line, = plt.plot(
                mean_x,
                mean_y,
                marker="o",
                linewidth=2,
                label=label,
            )

            plt.fill_between(
                mean_x,
                mean_y - std_y,
                mean_y + std_y,
                color=line.get_color(),
                alpha=0.15,
            )

    plt.xticks(sizes)

    plt.xlabel("Hidden layer size")
    plt.ylabel(ylabel)

    plt.title(
        f"{ylabel} — all trained models"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.title(textwrap.fill(plt.gca().get_title(), width=55))
    plt.ylabel(textwrap.fill(plt.gca().get_ylabel(), width=35))
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=250,
    )

    plt.close()


def plot_summary_metric(
    grouped,
    metric: str,
    ylabel: str,
    filename: str,
):
    sizes = sorted(grouped.keys())

    plt.figure(figsize=(11, 7))

    for key, label in SERIES_ORDER:
        if metric == "mean_steps" and "qat" in key:
            continue

        x = []
        y = []
        yerr = []

        for size in sizes:
            row = grouped[size].get(key)

            if row is None:
                continue

            x.append(size)

            if metric == "mean_reward":
                y.append(
                    to_float(
                        row,
                        "mean_of_mean_rewards",
                    )
                )

                yerr.append(
                    to_float(
                        row,
                        "std_across_models",
                    )
                )

            else:
                y.append(
                    to_float(row, metric)
                )

                yerr.append(0.0)

        if x:
            y = np.array(y)
            yerr = np.array(yerr)

            line, = plt.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                label=label,
            )

            plt.fill_between(
                x,
                y - yerr,
                y + yerr,
                color=line.get_color(),
                alpha=0.15,
            )

    plt.xticks(sizes)

    plt.xlabel("Hidden layer size")
    plt.ylabel(ylabel)

    plt.title(
        f"Summary {ylabel} vs model size"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.title(textwrap.fill(plt.gca().get_title(), width=55))
    plt.ylabel(textwrap.fill(plt.gca().get_ylabel(), width=35))
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=250,
    )

    plt.close()


def plot_summary_retention(grouped):
    sizes = sorted(grouped.keys())

    plt.figure(figsize=(11, 7))

    retention_series = [
        ("weight_qat", "W8 QAT"),
        ("weight_int8", "W8 exported"),
        ("weight_qat_int4", "W4 QAT"),
        ("weight_int4", "W4 exported"),
        (
            "weight_activation_qat",
            "W8A8 QAT",
        ),
        (
            "weight_activation_int8",
            "W8A8 exported",
        ),
        (
            "weight_activation_qat_int4",
            "W4A4 QAT",
        ),
        (
            "weight_activation_int4",
            "W4A4 exported",
        ),
    ]

    for key, label in retention_series:
        x = []
        y = []

        for size in sizes:
            base = grouped[size].get("base")
            other = grouped[size].get(key)

            if base is None or other is None:
                continue

            base_reward = to_float(
                base,
                "mean_of_mean_rewards",
            )

            other_reward = to_float(
                other,
                "mean_of_mean_rewards",
            )

            if base_reward == 0:
                continue

            retention = (
                100.0
                * other_reward
                / base_reward
            )

            x.append(size)
            y.append(retention)

        if x:
            plt.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                label=label,
            )

    plt.axhline(
        100.0,
        linestyle="--",
        linewidth=1,
    )

    plt.xticks(sizes)

    plt.xlabel("Hidden layer size")
    plt.ylabel("Reward retention (%)")

    plt.title(
        "Reward retention relative to FP32"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.title(textwrap.fill(plt.gca().get_title(), width=55))
    plt.ylabel(textwrap.fill(plt.gca().get_ylabel(), width=35))
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "summary_reward_retention.png",
        dpi=250,
    )

    plt.close()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    detailed_rows = load_rows(
        DETAILED_CSV_PATH
    )

    summary_rows = load_rows(
        SUMMARY_CSV_PATH
    )

    detailed_grouped = group_detailed_rows(
        detailed_rows
    )

    summary_grouped = group_summary_rows(
        summary_rows
    )

    # -------------------------------
    # Detailed plots
    # -------------------------------

    plot_detailed_metric(
        detailed_grouped,
        "mean_reward",
        "Mean reward",
        "detailed_mean_reward_all_versions.png",
    )

    if has_column(detailed_rows, "solve_rate"):
        plot_detailed_metric(
            detailed_grouped,
            "solve_rate",
            "Solve rate",
            "detailed_solve_rate_all_versions.png",
        )

    plot_detailed_metric(
        detailed_grouped,
        "success_rate",
        "Successful landing rate",
        "detailed_success_rate_all_versions.png",
    )

    if has_column(detailed_rows, "failure_rate"):
        plot_detailed_metric(
            detailed_grouped,
            "failure_rate",
            "Failure rate",
            "detailed_failure_rate_all_versions.png",
        )
    elif has_column(detailed_rows, "crash_rate"):
        plot_detailed_metric(
            detailed_grouped,
            "crash_rate",
            "Crash rate",
            "detailed_crash_rate_all_versions.png",
        )

    plot_detailed_metric(
        detailed_grouped,
        "mean_steps",
        "Mean episode steps",
        "detailed_mean_steps_all_versions.png",
    )

    # -------------------------------
    # Summary plots
    # -------------------------------

    plot_summary_metric(
        summary_grouped,
        "mean_reward",
        "Mean reward",
        "summary_mean_reward.png",
    )

    if has_column(summary_rows, "mean_solve_rate"):
        plot_summary_metric(
            summary_grouped,
            "mean_solve_rate",
            "Mean solve rate",
            "summary_solve_rate.png",
        )

    plot_summary_metric(
        summary_grouped,
        "mean_success_rate",
        "Mean successful landing rate",
        "summary_success_rate.png",
    )

    if has_column(summary_rows, "mean_failure_rate"):
        plot_summary_metric(
            summary_grouped,
            "mean_failure_rate",
            "Mean failure rate",
            "summary_failure_rate.png",
        )
    elif has_column(summary_rows, "mean_crash_rate"):
        plot_summary_metric(
            summary_grouped,
            "mean_crash_rate",
            "Mean crash rate",
            "summary_crash_rate.png",
        )

    plot_summary_metric(
        summary_grouped,
        "mean_steps",
        "Mean episode steps",
        "summary_mean_steps.png",
    )

    plot_summary_retention(
        summary_grouped
    )

    print(
        f"Saved plots to "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
