from LunarLander.paths import APP_ROOT
import argparse
from pathlib import Path

from LunarLander.deployment.esp_benchmark_data import (
    aggregate_results,
    load_rows,
)


DEFAULT_INPUT = Path(f"{APP_ROOT}/results/data/esp_benchmark_results.csv")
DEFAULT_OUTPUT = Path(f"{APP_ROOT}/results/tables/esp_benchmark_all_sizes.tex")


def format_row(row):
    values = [
        f"{row['size']}x{row['size']}",
        row["model_label"],
        str(row["episodes"]),
        f"{row['mean_reward']:.2f}",
        f"{row['std_reward']:.2f}",
        f"{row['solve_rate'] * 100.0:.2f}",
        f"{row['success_rate'] * 100.0:.2f}",
        f"{row['failure_rate'] * 100.0:.2f}",
        f"{row['device_inference_mean_ms']:.3f}",
        f"{row['serial_roundtrip_mean_ms']:.3f}",
    ]

    return " & ".join(values) + r" \\"


def build_table(rows):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        (
            r"\caption{ESP32 reward and inference performance across model "
            r"sizes. Repeated benchmark runs for the same model are combined.}"
        ),
        r"\label{tab:esp-benchmark-all}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        (
            r"Size & Model & Episodes & Mean reward $\uparrow$ & Reward SD "
            r"$\downarrow$ & Solve (\%) $\uparrow$ & Success (\%) $\uparrow$ & "
            r"Failure (\%) $\downarrow$ & ESP inference (ms) $\downarrow$ & "
            r"Round trip (ms) $\downarrow$ \\"
        ),
        r"\midrule",
    ]

    previous_size = None

    for row in rows:
        if previous_size is not None and row["size"] != previous_size:
            lines.append(r"\midrule")

        lines.append(format_row(row))
        previous_size = row["size"]

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table}",
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table from ESP32 benchmark results."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = aggregate_results(
        load_rows(args.input)
    )
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        build_table(rows)
    )

    print(f"Saved ESP32 LaTeX table to {args.output.resolve()}")


if __name__ == "__main__":
    main()
