"""Generate scratch-training reward tables using the existing thesis table style."""

from LunarLander.paths import APP_ROOT

import argparse
import csv
from pathlib import Path

from LunarLander.reporting.generate_table_data import LATEX_TABLE_SETTINGS, write_latex

ROOT = APP_ROOT
LABELS = {
    "fp32_scratch": "FP32 (Baseline)",
    "weight_qat_scratch": "W8 QAT",
    "int8_weight_scratch": "W8 exported",
}


def write_scratch_tables(summary, output_dir):
    """Use the same formatting, ranking, rounding and columns as other reward tables."""
    settings = dict(LATEX_TABLE_SETTINGS["mean_reward"])
    settings.update(
        caption="Mean reward comparison for FP32 and weight-only 8-bit QAT trained "
                "from scratch for 2.5 million steps. Reward SD, best reward and worst "
                "reward describe policy mean rewards across training seeds. Green and "
                "blue indicate the best and second-best values within each model size, respectively.",
        label="tab:scratch-mean-reward-comparison",
    )
    LATEX_TABLE_SETTINGS["scratch_mean_reward"] = settings
    rows = []
    for row in sorted(summary, key=lambda row: (int(row["size"]), list(LABELS).index(row["model_type"]))):
        size = int(row["size"])
        rows.append({
            "Size": f"{size}x{size}",
            "Model": LABELS[row["model_type"]],
            "Mean reward": f"{float(row['mean_of_mean_rewards']):.2f}",
            "Reward SD across seeds": f"{float(row['std_across_models']):.2f}",
            "Best mean reward": f"{float(row['best_mean_reward']):.2f}",
            "Worst mean reward": f"{float(row['worst_mean_reward']):.2f}",
            "Solve rate (%)": f"{100 * float(row['mean_solve_rate']):.2f}",
            "Success rate (%)": f"{100 * float(row['mean_success_rate']):.2f}",
            "Failure rate (%)": f"{100 * float(row['mean_failure_rate']):.2f}",
        })
    if not rows:
        raise ValueError("No scratch comparison results found")
    output_dir.mkdir(parents=True, exist_ok=True)
    largest_size = max(int(row["size"]) for row in summary)
    for suffix, selected in (
        ("all_sizes", rows),
        ("largest_size", [row for row in rows if row["Size"] == f"{largest_size}x{largest_size}"]),
    ):
        name = f"scratch_mean_reward_{suffix}"
        write_latex(output_dir / f"{name}.tex", name, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=APP_ROOT / "results/scratch/model_results_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "results/scratch")
    args = parser.parse_args()
    with args.input.open(newline="") as handle:
        summary = list(csv.DictReader(handle))
    write_scratch_tables(summary, args.output_dir)
    print(f"Saved scratch reward LaTeX tables to {args.output_dir}")


if __name__ == "__main__":
    main()
