from LunarLander.paths import APP_ROOT
import csv
from collections import defaultdict
from pathlib import Path


MODEL_LABELS = {
    "float32": "FP32",
    "int8": "INT8 (W8)",
}

MODEL_ORDER = [
    "float32",
    "int8",
]


def load_rows(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ESP32 benchmark results not found: {csv_path}. "
            "Run make esp_benchmark first."
        )

    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError(f"No benchmark rows found in {csv_path}")

    return rows


def pooled_stats(rows, mean_key, std_key, weight_key):
    weighted_rows = [
        (
            float(row[weight_key]),
            float(row[mean_key]),
            float(row[std_key]),
        )
        for row in rows
        if float(row[weight_key]) > 0
    ]

    total_weight = sum(weight for weight, _, _ in weighted_rows)

    if total_weight == 0:
        return 0.0, 0.0

    combined_mean = sum(
        weight * mean
        for weight, mean, _ in weighted_rows
    ) / total_weight

    combined_variance = sum(
        weight * (
            std ** 2
            + (mean - combined_mean) ** 2
        )
        for weight, mean, std in weighted_rows
    ) / total_weight

    return combined_mean, combined_variance ** 0.5


def weighted_mean(rows, value_key, weight_key):
    total_weight = sum(float(row[weight_key]) for row in rows)

    if total_weight == 0:
        return 0.0

    return sum(
        float(row[weight_key]) * float(row[value_key])
        for row in rows
    ) / total_weight


def aggregate_results(rows):
    grouped = defaultdict(list)

    for row in rows:
        key = (
            row["model_type"],
            int(row["size"]),
            int(row["train_seed"]),
        )
        grouped[key].append(row)

    aggregated = []

    for (model_type, size, train_seed), group_rows in grouped.items():
        reward_mean, reward_std = pooled_stats(
            group_rows,
            "mean_reward",
            "std_reward",
            "episodes",
        )
        step_mean, step_std = pooled_stats(
            group_rows,
            "mean_steps",
            "std_steps",
            "episodes",
        )
        device_mean, device_std = pooled_stats(
            group_rows,
            "device_inference_mean_ms",
            "device_inference_std_ms",
            "total_steps",
        )
        roundtrip_mean, roundtrip_std = pooled_stats(
            group_rows,
            "serial_roundtrip_mean_ms",
            "serial_roundtrip_std_ms",
            "total_steps",
        )

        aggregated.append({
            "model_type": model_type,
            "model_label": MODEL_LABELS.get(model_type, model_type),
            "size": size,
            "train_seed": train_seed,
            "runs": len(group_rows),
            "episodes": sum(int(row["episodes"]) for row in group_rows),
            "total_steps": sum(int(row["total_steps"]) for row in group_rows),
            "mean_reward": reward_mean,
            "std_reward": reward_std,
            "min_reward": min(float(row["min_reward"]) for row in group_rows),
            "max_reward": max(float(row["max_reward"]) for row in group_rows),
            "mean_steps": step_mean,
            "std_steps": step_std,
            "solve_rate": weighted_mean(group_rows, "solve_rate", "episodes"),
            "success_rate": weighted_mean(group_rows, "success_rate", "episodes"),
            "failure_rate": weighted_mean(group_rows, "failure_rate", "episodes"),
            "device_inference_mean_ms": device_mean,
            "device_inference_std_ms": device_std,
            "device_inference_min_ms": min(
                float(row["device_inference_min_ms"])
                for row in group_rows
            ),
            "device_inference_max_ms": max(
                float(row["device_inference_max_ms"])
                for row in group_rows
            ),
            "serial_roundtrip_mean_ms": roundtrip_mean,
            "serial_roundtrip_std_ms": roundtrip_std,
            "serial_roundtrip_min_ms": min(
                float(row["serial_roundtrip_min_ms"])
                for row in group_rows
            ),
            "serial_roundtrip_max_ms": max(
                float(row["serial_roundtrip_max_ms"])
                for row in group_rows
            ),
        })

    order = {
        model_type: index
        for index, model_type in enumerate(MODEL_ORDER)
    }
    aggregated.sort(
        key=lambda row: (
            row["size"],
            order.get(row["model_type"], len(order)),
            row["train_seed"],
        )
    )

    return aggregated
