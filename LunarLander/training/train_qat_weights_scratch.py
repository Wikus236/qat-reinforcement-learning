"""Train W8-only LunarLander QAT from random weights using base.py's protocol."""

from LunarLander.paths import APP_ROOT

import argparse
import csv
import json
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from LunarLander.training.train_base_initial import ENV_NAME, EVAL_SEEDS, MODEL_SIZES, SEEDS, evaluate_fixed_seeds, format_time
from rl_models.qat_weights import DQN_QAT_W

ROOT = APP_ROOT
POLICY_DIR = APP_ROOT / "models/qat_w8_scratch"
TOTAL_STEPS = 2_500_000
BLOCK_SIZE = 25_000
MODEL_TYPE = "weight_qat_scratch"


def train_one(size, seed, output_dir, total_steps=TOTAL_STEPS,
              block_size=BLOCK_SIZE, overwrite=False):
    """Match initial FP32 training, including blockwise learn/exploration behavior."""
    save_path = output_dir / f"qat_dqn_lander_{size}_{size}_seed_{seed}"
    curve_path = output_dir / "training_curves" / f"{MODEL_TYPE}_{size}_{size}_seed_{seed}.csv"
    checkpoint_dir = output_dir / "checkpoints" / f"{size}_{size}_seed_{seed}"
    manifest_path = output_dir / "runs" / f"{size}_{size}_seed_{seed}.json"
    int_path = Path(f"{save_path}_int8_weights.pt")
    config = dict(model_type=MODEL_TYPE, size=size, seed=seed,
                  total_steps=total_steps, block_size=block_size, from_scratch=True)
    if manifest_path.exists() and not overwrite:
        saved = json.loads(manifest_path.read_text())
        if (all(saved.get(key) == value for key, value in config.items())
                and saved.get("completed") and Path(f"{save_path}.zip").exists()
                and int_path.exists() and curve_path.exists()):
            print(f"Skipping completed {size}x{size}, seed {seed}")
            return
    existing = [Path(f"{save_path}.zip"), int_path, curve_path, manifest_path]
    if not overwrite and (any(path.exists() for path in existing)
                          or (checkpoint_dir.exists() and any(checkpoint_dir.iterdir()))):
        raise FileExistsError(f"Partial or incompatible run for {size}x{size}, seed {seed}. "
                              "Use --overwrite to restart this run from scratch.")
    for directory in (output_dir, curve_path.parent, checkpoint_dir, manifest_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({**config, "completed": False}, indent=2) + "\n")

    # The legacy learners seed inside learn(), after network construction.
    # Seed here too so each new scratch network has a reproducible initialization.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make(ENV_NAME, continuous=False, gravity=-10.0, enable_wind=False)
    try:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        model = DQN_QAT_W(
            "MlpPolicy", env, policy_kwargs=dict(net_arch=[size, size]),
            learning_rate=1e-4, buffer_size=100_000, learning_starts=10_000,
            batch_size=64, gamma=0.99, train_freq=4, gradient_steps=1,
            target_update_interval=1_000, exploration_fraction=0.2,
            exploration_final_eps=0.05, seed=seed, verbose=1,
        )
        # No checkpoint or pretrained FP32 weights are loaded.
        start = time.time()
        best_step = 0
        with curve_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model_type", "size", "train_seed", "training_steps",
                             "mean_reward", "std_reward"])
            mean, std = evaluate_fixed_seeds(model, env, EVAL_SEEDS)
            writer.writerow([MODEL_TYPE, size, seed, 0, mean, std])
            handle.flush()
            best_reward = mean
            model.save(save_path)
            model.save_int8_weights(int_path)
            for block in range(1, total_steps // block_size + 1):
                # Deliberately use the same block calls as base.py. Changing the
                # exploration horizon here would change the original experiment.
                model.learn(total_timesteps=block_size, log_interval=1,
                            reset_num_timesteps=False)
                mean, std = evaluate_fixed_seeds(model, env, EVAL_SEEDS)
                writer.writerow([MODEL_TYPE, size, seed, model.num_timesteps, mean, std])
                handle.flush()
                checkpoint = checkpoint_dir / (
                    f"weight_qat_scratch_block_{block:03d}_"
                    f"steps_{model.num_timesteps}_reward_{mean:.2f}")
                model.save(checkpoint)
                if mean > best_reward:
                    best_reward, best_step = mean, model.num_timesteps
                    model.save(save_path)
                    model.save_int8_weights(int_path)
                print(f"{size}x{size}, seed {seed}: {model.num_timesteps:,}/{total_steps:,}, "
                      f"reward {mean:.2f} +/- {std:.2f}, best {best_reward:.2f}, "
                      f"elapsed {format_time(time.time() - start)}", flush=True)
        manifest_path.write_text(json.dumps({
            **config, "completed": True, "best_step": best_step,
            "best_mean_reward": best_reward, "eval_seeds": EVAL_SEEDS,
        }, indent=2) + "\n")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, choices=MODEL_SIZES, default=MODEL_SIZES)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--output-dir", type=Path, default=POLICY_DIR)
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--overwrite", action="store_true", help="Restart selected runs, replacing outputs")
    args = parser.parse_args()
    if args.total_steps <= 0 or args.block_size <= 0 or args.total_steps % args.block_size:
        parser.error("total-steps must be positive and a multiple of positive block-size")
    if (args.total_steps, args.block_size) != (TOTAL_STEPS, BLOCK_SIZE) and args.output_dir.resolve() == POLICY_DIR:
        parser.error("Use a separate --output-dir for nonstandard/smoke runs")
    for size in args.sizes:
        for seed in args.seeds:
            train_one(size, seed, args.output_dir, args.total_steps, args.block_size, args.overwrite)


if __name__ == "__main__":
    main()
