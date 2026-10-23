from LunarLander.paths import APP_ROOT
import csv
from collections import defaultdict
from pathlib import Path


INPUT_CSV = Path(f"{APP_ROOT}/results/data/model_benchmark_results.csv")
OUTPUT_DIR = Path(f"{APP_ROOT}/results/tables")

MODEL_ORDER = [
    ("base_float", "FP32 (Baseline)", "Custom FP32 inference"),
    ("int8_weight_only", "W8 exported", "Measured INT8 weight export"),
    ("int8_weight_activation", "W8A8 exported", "Measured INT8 weights; theoretical INT8 activations"),
    ("int4_weight_only", "W4 simulated", "Simulated 4-bit weights stored as INT8; packed INT4 memory is theoretical"),
    ("int4_weight_activation", "W4A4 simulated", "Simulated 4-bit weights/activations stored as INT8; packed INT4 memory is theoretical"),
]

MODEL_LABELS = {
    model_type: label
    for model_type, label, _ in MODEL_ORDER
}

MODEL_NOTES = {
    model_type: note
    for model_type, _, note in MODEL_ORDER
}


def read_rows():
    with INPUT_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def require_new_terminal_rate_columns(rows):
    if not rows:
        return

    required_columns = {
        "solve_rate",
        "success_rate",
        "failure_rate",
    }
    missing_columns = [
        column
        for column in required_columns
        if column not in rows[0]
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise SystemExit(
            "model_benchmark_results.csv does not contain the new "
            f"terminal-rate columns: {missing}. Rerun benchmark.py first."
        )


def mean(values):
    values = list(values)
    if not values:
        return 0.0

    return sum(values) / len(values)


def std(values):
    values = list(values)
    if len(values) <= 1:
        return 0.0

    avg = mean(values)
    variance = sum(
        (value - avg) ** 2
        for value in values
    ) / len(values)

    return variance ** 0.5


def theoretical_activation_memory_mb(size, model_type):
    hidden_activation_elements = size * 2

    if model_type == "int8_weight_activation":
        bytes_per_activation = 1.0
    elif model_type == "int4_weight_activation":
        bytes_per_activation = 0.5
    else:
        bytes_per_activation = 4.0

    return (
        hidden_activation_elements
        * bytes_per_activation
        / (1024 ** 2)
    )


def value(row, key):
    raw = row.get(key, "")

    if raw == "":
        if key == "solve_rate":
            # Old benchmark files used success_rate for the reward >= 200 metric.
            return value(row, "success_rate")

        if key == "theoretical_activation_memory_mb":
            return theoretical_activation_memory_mb(
                int(row["size"]),
                row["model_type"],
            )

        if key == "theoretical_runtime_memory_mb":
            return (
                value(row, "theoretical_parameter_memory_mb")
                + value(row, "theoretical_activation_memory_mb")
            )

    return float(raw)


def reduction_percent(base_value, model_value):
    if base_value == 0:
        return 0.0

    return (
        (base_value - model_value)
        / base_value
        * 100.0
    )


def grouped_rows(rows):
    grouped = defaultdict(list)

    for row in rows:
        model_type = row["model_type"]

        if model_type not in MODEL_LABELS:
            continue

        grouped[(int(row["size"]), model_type)].append(row)

    return grouped


def model_types_present(grouped, size):
    present = {
        model_type
        for row_size, model_type in grouped
        if row_size == size
    }

    return [
        model_type
        for model_type, _, _ in MODEL_ORDER
        if model_type in present
    ]


def format_number(value, decimals=6):
    return f"{value:.{decimals}f}"


def format_percent(value):
    return f"{value:.2f}"


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


LATEX_TABLE_SETTINGS = {
    "mean_reward": {
        "caption": "Mean reward comparison across model sizes. Green and blue indicate the best and second-best values within each model size, respectively.",
        "label": "tab:mean-reward-comparison",
        "columns": [
            "Size",
            "Model",
            "Mean reward $\\uparrow$",
            "Reward SD $\\downarrow$",
            "Best reward $\\uparrow$",
            "Worst reward $\\uparrow$",
            "Solve (\\%) $\\uparrow$",
            "Success (\\%) $\\uparrow$",
            "Failure (\\%) $\\downarrow$",
        ],
        "source_fields": [
            "Size",
            "Model",
            "Mean reward",
            "Reward SD across seeds",
            "Best mean reward",
            "Worst mean reward",
            "Solve rate (%)",
            "Success rate (%)",
            "Failure rate (%)",
        ],
        "alignment": "llrrrrrrr",
        "rank_directions": {
            "Mean reward": "max",
            "Reward SD across seeds": "min",
            "Best mean reward": "max",
            "Worst mean reward": "max",
            "Solve rate (%)": "max",
            "Success rate (%)": "max",
            "Failure rate (%)": "min",
        },
    },
    "memory_reduction": {
        "caption": "Memory comparison across model sizes. Actual stored parameter memory reflects the current tensor representation, while theoretical values assume efficient packed quantized deployment. Green and blue indicate the best and second-best reductions within each model size, respectively.",
        "label": "tab:memory-comparison",
        "columns": [
            "Size",
            "Model",
            "Actual params $\\downarrow$",
            "Actual red. $\\uparrow$",
            "Theor. params $\\downarrow$",
            "Theor. red. $\\uparrow$",
            "Theor. acts. $\\downarrow$",
            "Act. red. $\\uparrow$",
            "Theor. runtime $\\downarrow$",
            "Runtime red. $\\uparrow$",
        ],
        "units": [
            "",
            "",
            "(MB)",
            "(\\%)",
            "(MB)",
            "(\\%)",
            "(MB)",
            "(\\%)",
            "(MB)",
            "(\\%)",
        ],
        "source_fields": [
            "Size",
            "Model",
            "Actual stored params (MB)",
            "Actual stored reduction (%)",
            "Theoretical packed params (MB)",
            "Theoretical param reduction (%)",
            "Theoretical activations (MB)",
            "Activation reduction (%)",
            "Theoretical runtime memory (MB)",
            "Runtime memory reduction (%)",
        ],
        "alignment": "llrrrrrrrr",
        "rank_directions": {
            "Actual stored reduction (%)": "max",
            "Theoretical param reduction (%)": "max",
            "Runtime memory reduction (%)": "max",
        },
        "post_note": "The simulated INT4 models restrict values to a 4-bit range, but the current implementation stores these values using INT8 tensors. Therefore, actual stored parameter memory reflects the current INT8-backed representation, while theoretical packed parameter and runtime memory assume an efficient packed INT4 deployment.",
    },
    "inference_time": {
        "caption": "Deployment-style NumPy inference time comparison across model sizes. Green and blue indicate the best and second-best values within each model size, respectively.",
        "label": "tab:inference-time-comparison",
        "columns": [
            "Size",
            "Model",
            "Mean time $\\downarrow$",
            "SD across seeds $\\downarrow$",
            "Mean timing SD $\\downarrow$",
            "Reduction $\\uparrow$",
            "Note",
        ],
        "units": [
            "",
            "",
            "($\\mu$s)",
            "($\\mu$s)",
            "($\\mu$s)",
            "(\\%)",
            "",
        ],
        "source_fields": [
            "Size",
            "Model",
            "Mean inference time (us)",
            "Inference SD across seeds (us)",
            "Mean timing SD within benchmark (us)",
            "Inference time reduction vs FP32 (%)",
            "Note",
        ],
        "alignment": "llrrrrl",
        "rank_directions": {
            "Mean inference time (us)": "min",
            "Inference time reduction vs FP32 (%)": "max",
        },
        "post_note": "All displayed models use the same custom layer-by-layer NumPy inference structure. Quantized weights are dequantized for floating-point matrix operations because native low-precision kernels are not used. Simulated INT4 values remain stored in INT8 arrays.",
    },
}

SHORT_NOTES = {
    "Custom FP32 inference": "Custom FP32",
    "Measured INT8 weight export": "INT8 weights",
    "Measured INT8 weights; theoretical INT8 activations": "INT8 weights, INT8 activations",
    "Simulated 4-bit weights stored as INT8; packed INT4 memory is theoretical": "Simulated W4, stored as INT8",
    "Simulated 4-bit weights/activations stored as INT8; packed INT4 memory is theoretical": "Simulated W4A4, stored as INT8",
}


def ranking_lookup(rows, rank_directions):
    lookup = {}
    sizes = {
        row.get("Size")
        for row in rows
    }

    for size in sizes:
        size_rows = [
            row
            for row in rows
            if row.get("Size") == size
        ]

        for field, direction in rank_directions.items():
            values = sorted(
                {float(row[field]) for row in size_rows},
                reverse=direction == "max",
            )

            for rank, value in enumerate(values[:2]):
                lookup[(size, field, value)] = rank

    return lookup


def latex_escape(value):
    text = str(value)
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    return text


def latex_note(value):
    return SHORT_NOTES.get(
        str(value),
        str(value),
    )


def table_kind(name):
    for kind in LATEX_TABLE_SETTINGS:
        if name.startswith(kind):
            return kind

    raise ValueError(
        f"No LaTeX table settings for {name}"
    )


def latex_label(base_label, name):
    if name.endswith("_largest_size"):
        return f"{base_label}-largest"

    return f"{base_label}-all"


def write_latex(path, name, rows):
    settings = LATEX_TABLE_SETTINGS[
        table_kind(name)
    ]
    columns = settings["columns"]
    units = settings.get("units")
    source_fields = settings["source_fields"]
    alignment = settings["alignment"]
    rank_directions = settings.get("rank_directions", {})
    ranks = ranking_lookup(rows, rank_directions)
    label = latex_label(
        settings["label"],
        name,
    )

    with path.open("w") as handle:
        handle.write("% Requires: \\usepackage{pdflscape}, \\usepackage{booktabs}, \\usepackage{adjustbox}, \\usepackage{colortbl}\n")
        handle.write("% Define in the preamble: \\newcommand{\\best}[1]{\\cellcolor{green!25}\\textbf{#1}}\n")
        handle.write("% Define in the preamble: \\newcommand{\\secondbest}[1]{\\cellcolor{blue!15}#1}\n")
        handle.write("\\begin{landscape}\n")
        handle.write("\\begin{table}[htbp]\n")
        handle.write("\\centering\n")
        handle.write("\\small\n")
        handle.write(f"\\caption{{{latex_escape(settings['caption'])}}}\n")
        handle.write(f"\\label{{{label}}}\n")
        handle.write("\\begin{adjustbox}{max width=\\linewidth}\n")
        handle.write(f"\\begin{{tabular}}{{{alignment}}}\n")
        handle.write("\\toprule\n")
        handle.write(
            " & ".join(columns)
            + r" \\" + "\n"
        )

        if units:
            handle.write(
                " & ".join(units)
                + r" \\" + "\n"
            )

        handle.write("\\midrule\n")

        previous_size = None

        for row in rows:
            current_size = row.get("Size")

            if previous_size is not None and current_size != previous_size:
                handle.write("\\midrule\n")

            cells = []

            for field in source_fields:
                value = row[field]

                if field == "Note":
                    value = latex_note(value)

                cell = latex_escape(value)

                if field in rank_directions:
                    rank = ranks.get(
                        (current_size, field, float(value))
                    )
                    if rank == 0:
                        cell = f"\\best{{{cell}}}"
                    elif rank == 1:
                        cell = f"\\secondbest{{{cell}}}"

                cells.append(cell)

            handle.write(
                " & ".join(cells)
                + r" \\" + "\n"
            )
            previous_size = current_size

        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")
        handle.write("\\end{adjustbox}\n")
        handle.write("\\end{table}\n")
        handle.write("\\end{landscape}\n")

        if settings.get("post_note"):
            handle.write("\n\\noindent\n")
            handle.write(latex_escape(settings["post_note"]))
            handle.write("\n")


def write_table(name, rows, fieldnames):
    write_csv(
        OUTPUT_DIR / f"{name}.csv",
        rows,
        fieldnames,
    )
    write_latex(
        OUTPUT_DIR / f"{name}.md",
        name,
        rows,
    )
    write_latex(
        OUTPUT_DIR / f"{name}.tex",
        name,
        rows,
    )


def build_reward_rows(grouped, sizes):
    table_rows = []

    for size in sizes:
        for model_type in model_types_present(grouped, size):
            rows = grouped[(size, model_type)]
            rewards = [
                value(row, "mean_reward")
                for row in rows
            ]
            solve_rates = [
                value(row, "solve_rate")
                for row in rows
            ]
            success_rates = [
                value(row, "success_rate")
                for row in rows
            ]
            failure_rates = [
                value(row, "failure_rate")
                for row in rows
            ]

            table_rows.append({
                "Size": f"{size}x{size}",
                "Model": MODEL_LABELS[model_type],
                "Mean reward": format_number(mean(rewards), 2),
                "Reward SD across seeds": format_number(std(rewards), 2),
                "Best mean reward": format_number(max(rewards), 2),
                "Worst mean reward": format_number(min(rewards), 2),
                "Solve rate (%)": format_percent(mean(solve_rates) * 100.0),
                "Success rate (%)": format_percent(mean(success_rates) * 100.0),
                "Failure rate (%)": format_percent(mean(failure_rates) * 100.0),
            })

    return table_rows


def build_memory_rows(grouped, sizes):
    table_rows = []

    metrics = [
        "parameter_memory_mb",
        "theoretical_parameter_memory_mb",
        "theoretical_activation_memory_mb",
        "theoretical_runtime_memory_mb",
    ]

    for size in sizes:
        base_rows = grouped.get((size, "base_float"), [])

        if not base_rows:
            continue

        base_values = {
            metric: mean(
                value(row, metric)
                for row in base_rows
            )
            for metric in metrics
        }

        for model_type in model_types_present(grouped, size):
            rows = grouped[(size, model_type)]
            averages = {
                metric: mean(
                    value(row, metric)
                    for row in rows
                )
                for metric in metrics
            }

            table_rows.append({
                "Size": f"{size}x{size}",
                "Model": MODEL_LABELS[model_type],
                "Actual stored params (MB)": format_number(averages["parameter_memory_mb"]),
                "Actual stored reduction (%)": format_percent(
                    reduction_percent(
                        base_values["parameter_memory_mb"],
                        averages["parameter_memory_mb"],
                    )
                ),
                "Theoretical packed params (MB)": format_number(averages["theoretical_parameter_memory_mb"]),
                "Theoretical param reduction (%)": format_percent(
                    reduction_percent(
                        base_values["theoretical_parameter_memory_mb"],
                        averages["theoretical_parameter_memory_mb"],
                    )
                ),
                "Theoretical activations (MB)": format_number(averages["theoretical_activation_memory_mb"]),
                "Activation reduction (%)": format_percent(
                    reduction_percent(
                        base_values["theoretical_activation_memory_mb"],
                        averages["theoretical_activation_memory_mb"],
                    )
                ),
                "Theoretical runtime memory (MB)": format_number(averages["theoretical_runtime_memory_mb"]),
                "Runtime memory reduction (%)": format_percent(
                    reduction_percent(
                        base_values["theoretical_runtime_memory_mb"],
                        averages["theoretical_runtime_memory_mb"],
                    )
                ),
                "Note": MODEL_NOTES[model_type],
            })

    return table_rows


def build_inference_rows(grouped, sizes):
    table_rows = []

    for size in sizes:
        base_rows = grouped.get((size, "base_float"), [])

        if not base_rows:
            continue

        base_mean = mean(
            value(row, "inference_mean_ms")
            for row in base_rows
        )

        for model_type in model_types_present(grouped, size):
            rows = grouped[(size, model_type)]
            mean_times = [
                value(row, "inference_mean_ms")
                for row in rows
            ]
            timing_sds = [
                value(row, "inference_std_ms")
                for row in rows
            ]

            table_rows.append({
                "Size": f"{size}x{size}",
                "Model": MODEL_LABELS[model_type],
                "Mean inference time (us)": format_number(1000.0 * mean(mean_times), 6),
                "Inference SD across seeds (us)": format_number(1000.0 * std(mean_times), 6),
                "Mean timing SD within benchmark (us)": format_number(1000.0 * mean(timing_sds), 6),
                "Inference time reduction vs FP32 (%)": format_percent(
                    reduction_percent(
                        base_mean,
                        mean(mean_times),
                    )
                ),
                "Note": MODEL_NOTES[model_type],
            })

    return table_rows


def filter_largest_size(rows):
    sizes = sorted({
        int(row["Size"].split("x")[0])
        for row in rows
    })

    if not sizes:
        return []

    largest = f"{sizes[-1]}x{sizes[-1]}"

    return [
        row
        for row in rows
        if row["Size"] == largest
    ]


def main():
    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    rows = read_rows()
    require_new_terminal_rate_columns(rows)
    grouped = grouped_rows(rows)
    sizes = sorted({
        size
        for size, _ in grouped
    })

    reward_rows = build_reward_rows(
        grouped,
        sizes,
    )
    memory_rows = build_memory_rows(
        grouped,
        sizes,
    )
    inference_rows = build_inference_rows(
        grouped,
        sizes,
    )

    reward_fields = [
        "Size",
        "Model",
        "Mean reward",
        "Reward SD across seeds",
        "Best mean reward",
        "Worst mean reward",
        "Solve rate (%)",
        "Success rate (%)",
        "Failure rate (%)",
    ]
    memory_fields = [
        "Size",
        "Model",
        "Actual stored params (MB)",
        "Actual stored reduction (%)",
        "Theoretical packed params (MB)",
        "Theoretical param reduction (%)",
        "Theoretical activations (MB)",
        "Activation reduction (%)",
        "Theoretical runtime memory (MB)",
        "Runtime memory reduction (%)",
        "Note",
    ]
    inference_fields = [
        "Size",
        "Model",
        "Mean inference time (us)",
        "Inference SD across seeds (us)",
        "Mean timing SD within benchmark (us)",
        "Inference time reduction vs FP32 (%)",
        "Note",
    ]

    write_table(
        "mean_reward_all_sizes",
        reward_rows,
        reward_fields,
    )
    write_table(
        "memory_reduction_all_sizes",
        memory_rows,
        memory_fields,
    )
    write_table(
        "inference_time_all_sizes",
        inference_rows,
        inference_fields,
    )

    write_table(
        "mean_reward_largest_size",
        filter_largest_size(reward_rows),
        reward_fields,
    )
    write_table(
        "memory_reduction_largest_size",
        filter_largest_size(memory_rows),
        memory_fields,
    )
    write_table(
        "inference_time_largest_size",
        filter_largest_size(inference_rows),
        inference_fields,
    )

    print(
        f"Saved table data to {OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
