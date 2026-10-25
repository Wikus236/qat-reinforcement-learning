"""Check full training coverage and schedule before generating rerun reports."""

import csv
import json
import math
from pathlib import Path

import torch

from LunarLander.paths import APP_ROOT
from LunarLander.training.settings import (
    EXPLORATION_DECAY_STEPS, ADDITIONAL_EXPLORATION_INITIAL_EPS,
)
from LunarLander.training.train_base_initial import MODEL_SIZES, SEEDS


SPECS = (
    ("initial_fp32", "my_dqn_lander", "base", 2_500_000, None),
    ("extended_fp32", "my_dqn_lander", "base_extended", 500_000, None),
    ("qat_w8", "qat_dqn_lander", "weight_qat", 500_000, "int8"),
    ("qat_w8a8", "qat_dqn_lander", "weight_activation_qat", 500_000, "int8"),
    ("qat_w4", "qat_dqn_lander", "weight_qat_int4", 500_000, "int4"),
    ("qat_w4a4", "qat_dqn_lander", "weight_activation_qat_int4", 500_000, "int4"),
    ("qat_w8_scratch", "qat_dqn_lander", "weight_qat_scratch", 2_500_000, "int8"),
)


def validate_experiment(root=APP_ROOT):
    root = Path(root)
    errors = []
    checkpoint_count = curve_count = export_count = 0
    for folder, prefix, curve_prefix, budget, export in SPECS:
        initial_eps = 1.0 if budget == 2_500_000 else ADDITIONAL_EXPLORATION_INITIAL_EPS
        final_eps = 0.05 if budget == 2_500_000 else 0.01
        directory = root / "models" / folder
        for size in MODEL_SIZES:
            for seed in SEEDS:
                suffix = f"{size}_{size}_seed_{seed}"
                checkpoint_path = directory / f"{prefix}_{suffix}.zip"
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    expected = {"exploration_decay_steps": EXPLORATION_DECAY_STEPS,
                                "exploration_initial_eps": initial_eps,
                                "exploration_final_eps": final_eps}
                    if any(not math.isclose(checkpoint.get(key, float("nan")), value)
                           for key, value in expected.items()):
                        errors.append(f"Wrong exploration schedule: {checkpoint_path}")
                    if checkpoint.get("seed") != seed:
                        errors.append(f"Wrong training seed: {checkpoint_path}")
                    if checkpoint.get("policy_kwargs", {}).get("net_arch") != [size, size]:
                        errors.append(f"Wrong architecture: {checkpoint_path}")
                    checkpoint_count += 1
                except (OSError, ValueError, TypeError, RuntimeError, EOFError) as error:
                    errors.append(f"Cannot validate {checkpoint_path}: {error}")

                curve_path = directory / "training_curves" / f"{curve_prefix}_{suffix}.csv"
                try:
                    with curve_path.open(newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    if ([int(row["training_steps"]) for row in rows]
                            != list(range(0, budget + 1, 25_000))
                            or any(int(row["size"]) != size or int(row["train_seed"]) != seed
                                   or not math.isfinite(float(row["mean_reward"])) for row in rows)):
                        errors.append(f"Incomplete or mismatched training curve: {curve_path}")
                    curve_count += 1
                except (OSError, ValueError, KeyError) as error:
                    errors.append(f"Cannot validate {curve_path}: {error}")

                if export is not None:
                    path = directory / f"{prefix}_{suffix}_{export}_weights.pt"
                    if not path.is_file():
                        errors.append(f"Missing export: {path}")
                    export_count += 1

                if folder == "qat_w8_scratch":
                    path = directory / "runs" / f"{suffix}.json"
                    try:
                        state = json.loads(path.read_text())
                        if (state.get("completed") is not True
                                or state.get("exploration_decay_steps") != EXPLORATION_DECAY_STEPS
                                or state.get("total_steps") != budget):
                            errors.append(f"Incomplete or incompatible scratch run: {path}")
                    except (OSError, ValueError) as error:
                        errors.append(f"Cannot validate {path}: {error}")
    if errors:
        extra = f"\n... and {len(errors) - 30} more issues." if len(errors) > 30 else ""
        raise ValueError("Cannot generate reports from an incomplete or mixed experiment:\n"
                         + "\n".join(errors[:30]) + extra)
    for directory in ("data", "figures", "tables", "scratch"):
        (root / "results" / directory).mkdir(parents=True, exist_ok=True)
    print(f"Validated {checkpoint_count} checkpoints, {export_count} exports and "
          f"{curve_count} complete training curves.")


if __name__ == "__main__":
    try:
        validate_experiment()
    except ValueError as error:
        raise SystemExit(str(error)) from None
