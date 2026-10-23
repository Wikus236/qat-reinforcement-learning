"""Compare initial 2.5M-step FP32 training with W8-only QAT from scratch.

Reuses compare.py's exact 100 test seeds, deterministic evaluation and metrics.
Results and training plots are isolated from the existing comparison outputs.
"""

from LunarLander.paths import APP_ROOT

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from LunarLander.training.train_base_initial import ENV_NAME, MODEL_SIZES, SEEDS
from LunarLander.evaluation.compare import evaluate_model
from rl_models.inference import Float32QNetwork, Int8QNetwork
from rl_models.qat_weights import DQN_QAT_W
from LunarLander.reporting.generate_scratch_table import write_scratch_tables
from LunarLander.training.train_qat_weights_scratch import POLICY_DIR, TOTAL_STEPS, BLOCK_SIZE, MODEL_TYPE

ROOT = APP_ROOT
TYPES = ("fp32_scratch", "weight_qat_scratch", "int8_weight_scratch")


def read_curve(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_runs(fp32_dir, qat_dir, sizes, seeds):
    """Fail before evaluation if a full matched experiment is unavailable."""
    specs, curves, errors = [], {}, []
    for size in sizes:
        for seed in seeds:
            suffix = f"{size}_{size}_seed_{seed}"
            paths = (
                fp32_dir / f"my_dqn_lander_{suffix}.zip",
                qat_dir / f"qat_dqn_lander_{suffix}.zip",
                qat_dir / f"qat_dqn_lander_{suffix}_int8_weights.pt",
            )
            for kind, path in zip(TYPES, paths):
                if not path.is_file():
                    errors.append(f"Missing {path}")
                specs.append((kind, size, seed, path))
            manifest = qat_dir / "runs" / f"{suffix}.json"
            if not manifest.exists():
                errors.append(f"Missing completion record {manifest}")
            else:
                state = json.loads(manifest.read_text())
                expected = dict(completed=True, from_scratch=True, total_steps=TOTAL_STEPS,
                                block_size=BLOCK_SIZE, size=size, seed=seed, model_type=MODEL_TYPE)
                if any(state.get(key) != value for key, value in expected.items()):
                    errors.append(f"Incomplete or nonstandard QAT run: {manifest}")
            for kind, path in (
                (TYPES[0], fp32_dir / "training_curves" / f"base_{suffix}.csv"),
                (TYPES[1], qat_dir / "training_curves" / f"{MODEL_TYPE}_{suffix}.csv"),
            ):
                if not path.exists():
                    errors.append(f"Missing {path}")
                    continue
                rows = read_curve(path)
                expected_steps = list(range(0, TOTAL_STEPS + 1, BLOCK_SIZE))
                if ([int(row["training_steps"]) for row in rows] != expected_steps
                        or any(int(row["size"]) != size or int(row["train_seed"]) != seed for row in rows)):
                    errors.append(f"Curve does not match the original 2.5M-step protocol: {path}")
                curves[kind, size, seed] = rows
    if errors:
        raise ValueError("Cannot compare unmatched runs:\n" + "\n".join(errors))
    return specs, curves


def plot_curves(curves, sizes, seeds, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Match the typography used by plot_compare.py and plot_benchmark.py.
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 19,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 15,
        "figure.titlesize": 22,
    })

    combined = []
    all_fig, all_ax = plt.subplots(figsize=(14, 9))
    colors = plt.get_cmap("tab10")
    for size_index, size in enumerate(sizes):
        fig, ax = plt.subplots(figsize=(11, 7))
        for kind, label in ((TYPES[0], "FP32 from scratch"), (TYPES[1], "W8 QAT from scratch")):
            runs = [curves[kind, size, seed] for seed in seeds]
            steps = np.array([int(row["training_steps"]) for row in runs[0]])
            values = np.array([[float(row["mean_reward"]) for row in rows] for rows in runs])
            means, stds = values.mean(axis=0), values.std(axis=0)
            ax.plot(steps, means, label=label)
            ax.fill_between(steps, means - stds, means + stds, alpha=0.2)
            color = colors(size_index % 10)
            all_ax.plot(steps, means, color=color,
                        linestyle="-" if kind == TYPES[0] else "--",
                        label=f"{size}x{size} {'FP32' if kind == TYPES[0] else 'W8 QAT'}")
            all_ax.fill_between(steps, means - stds, means + stds,
                                color=color, alpha=0.07)
            for step, mean, std in zip(steps, means, stds):
                combined.append(dict(model_type=kind, size=size, training_steps=int(step),
                                     num_models=len(seeds), mean_reward=float(mean),
                                     std_across_seeds=float(std)))
        ax.axhline(200, linestyle="--", color="gray", label="Solve threshold")
        ax.set(xlabel="Training environment steps", ylabel="Fixed-seed mean reward",
               title=f"LunarLander {size}x{size}: mean +/- SD across {len(seeds)} seeds")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"training_curve_{size}_{size}.png", dpi=180)
        plt.close(fig)
    all_ax.axhline(200, linestyle=":", color="gray", label="Solve threshold")
    all_ax.set(xlabel="Training environment steps", ylabel="Fixed-seed mean reward",
               title=f"LunarLander all model sizes: mean +/- SD across {len(seeds)} seeds")
    all_ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    all_ax.grid(alpha=0.2)
    all_fig.tight_layout()
    all_fig.savefig(output_dir / "training_curve_all_sizes.png", dpi=180, bbox_inches="tight")
    plt.close(all_fig)
    write_rows(output_dir / "training_curves_summary.csv", combined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-dir", type=Path, default=APP_ROOT / "models/initial_fp32",
                        help="Initial base.py runs, NOT the extended policies_base runs")
    parser.add_argument("--qat-dir", type=Path, default=POLICY_DIR)
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "results/scratch")
    parser.add_argument("--sizes", type=int, nargs="+", choices=MODEL_SIZES, default=MODEL_SIZES)
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=SEEDS)
    parser.add_argument("--curves-only", action="store_true", help="Plot curves without final policy evaluation")
    args = parser.parse_args()
    try:
        specs, curves = validate_runs(args.fp32_dir, args.qat_dir, args.sizes, args.seeds)
    except ValueError as error:
        parser.exit(1, str(error) + "\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(curves, args.sizes, args.seeds, args.output_dir)
    if args.curves_only:
        return
    rows = []
    env = gym.make(ENV_NAME, continuous=False, gravity=-10.0, enable_wind=False)
    try:
        for kind, size, seed, path in specs:
            print(f"Evaluating {kind}, {size}x{size}, seed {seed}", flush=True)
            if kind == TYPES[0]:
                model = Float32QNetwork(path)
            elif kind == TYPES[1]:
                model = DQN_QAT_W.load(path, env=env)
            else:
                model = Int8QNetwork(path)
            rows.append(dict(model_type=kind, size=size, train_seed=seed,
                             model=str(path), **evaluate_model(model, env)))
    finally:
        env.close()
    write_rows(args.output_dir / "model_results_detailed.csv", rows)
    summary, paired = [], []
    for size in args.sizes:
        for kind in TYPES:
            group = [row for row in rows if row["size"] == size and row["model_type"] == kind]
            rewards = np.array([row["mean_reward"] for row in group])
            summary.append(dict(
                model_type=kind, size=size, num_models=len(group),
                mean_of_mean_rewards=float(rewards.mean()), std_across_models=float(rewards.std()),
                best_mean_reward=float(rewards.max()), worst_mean_reward=float(rewards.min()),
                mean_solve_rate=float(np.mean([row["solve_rate"] for row in group])),
                mean_success_rate=float(np.mean([row["success_rate"] for row in group])),
                mean_failure_rate=float(np.mean([row["failure_rate"] for row in group])),
                mean_steps=float(np.mean([row["mean_steps"] for row in group])),
            ))
        for seed in args.seeds:
            group = {row["model_type"]: row for row in rows if row["size"] == size and row["train_seed"] == seed}
            for kind in TYPES[1:]:
                paired.append(dict(size=size, train_seed=seed, model_type=kind,
                                   fp32_mean_reward=group[TYPES[0]]["mean_reward"],
                                   quantized_mean_reward=group[kind]["mean_reward"],
                                   reward_difference=group[kind]["mean_reward"] - group[TYPES[0]]["mean_reward"],
                                   solve_rate_difference=group[kind]["solve_rate"] - group[TYPES[0]]["solve_rate"]))
    write_rows(args.output_dir / "model_results_summary.csv", summary)
    write_rows(args.output_dir / "paired_differences.csv", paired)
    write_scratch_tables(summary, args.output_dir)
    print(f"Saved separate comparison results and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
