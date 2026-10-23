from LunarLander.paths import APP_ROOT
import gymnasium as gym
from pathlib import Path
from rl_models.dqn import DQN
import time
import csv
import numpy as np

from LunarLander.training.train_base_initial import MODEL_SIZES, SEEDS


ENV_NAME = "LunarLander-v3"


# Fixed seeds used for every evaluation.
# Ideally these should NOT overlap with the training seeds in SEEDS.
EVAL_SEEDS = [
    42,
    123,
    456,
    789,
    111,
    234,
    23,
    36,
    964,
    231,
]


def evaluate_fixed_seeds(model, env, seeds):
    """
    Evaluate a model on exactly the same environment seeds every time.

    This makes the training curves directly comparable between:
        - Base FP32
        - Weight-only QAT
        - Weight + activation INT8 QAT
        - Weight + activation INT4 QAT
    """

    rewards = []

    for eval_seed in seeds:
        obs, info = env.reset(seed=eval_seed)

        total_reward = 0.0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

        rewards.append(total_reward)

    rewards = np.asarray(
        rewards,
        dtype=np.float64,
    )

    return (
        float(np.mean(rewards)),
        float(np.std(rewards)),
    )


def format_time(seconds):
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"

    if minutes > 0:
        return f"{minutes}m {secs}s"

    return f"{secs}s"


if __name__ == "__main__":

    for size in MODEL_SIZES:

        for seed in SEEDS:

            # ---------------------------------------------------------
            # Paths
            # ---------------------------------------------------------

            # Original pretrained FP32 model
            load_path = Path(
                f"{APP_ROOT}/models/initial_fp32/"
                f"my_dqn_lander_{size}_{size}_seed_{seed}"
            )

            # Best model after additional FP32 training
            save_path = Path(
                f"{APP_ROOT}/models/extended_fp32/"
                f"my_dqn_lander_{size}_{size}_seed_{seed}"
            )

            # All intermediate checkpoints
            checkpoint_dir = Path(
                f"{APP_ROOT}/models/extended_fp32/checkpoints/"
                f"{size}_{size}_seed_{seed}"
            )

            # Learning curve CSV
            curve_path = Path(
                f"{APP_ROOT}/models/extended_fp32/training_curves/"
                f"base_extended_{size}_{size}_seed_{seed}.csv"
            )

            # ---------------------------------------------------------
            # Check original model exists
            # ---------------------------------------------------------

            if not Path(str(load_path) + ".zip").exists():

                print(
                    f"Skipping {size}x{size} seed {seed}: "
                    f"base model not found at {load_path}.zip"
                )

                continue

            # ---------------------------------------------------------
            # Create directories
            # ---------------------------------------------------------

            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            curve_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ---------------------------------------------------------
            # Environment
            # ---------------------------------------------------------

            env = gym.make(
                ENV_NAME,
                continuous=False,
                gravity=-10.0,
                enable_wind=False,
            )

            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            # ---------------------------------------------------------
            # Create model
            # ---------------------------------------------------------

            policy_kwargs = dict(
                net_arch=[size, size]
            )

            model = DQN(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=1e-5,
                buffer_size=200_000,
                learning_starts=20_000,
                batch_size=128,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=2_000,
                exploration_fraction=0.05,
                exploration_final_eps=0.01,
                seed=seed,
                verbose=1,
            )

            # Load weights from the original pretrained model
            model.load_float_model(load_path)

            # ---------------------------------------------------------
            # Additional training configuration
            # ---------------------------------------------------------

            block_size = 25_000
            total_timesteps = 500_000

            num_blocks = (
                total_timesteps // block_size
            )

            start_time = time.time()

            print()
            print("=" * 70)

            print(
                f"Starting additional FP32 training "
                f"for {size}x{size}, seed {seed}"
            )

            print(
                f"Loaded base model: "
                f"{load_path}.zip"
            )

            print(
                f"Total additional timesteps: "
                f"{total_timesteps:,}"
            )

            print(
                f"Block size: "
                f"{block_size:,}"
            )

            print(
                f"Total blocks: "
                f"{num_blocks}"
            )

            print(
                f"Best model save path: "
                f"{save_path}.zip"
            )

            print(
                f"Curve CSV: "
                f"{curve_path}"
            )

            print("=" * 70)

            # ---------------------------------------------------------
            # Open training curve CSV
            # ---------------------------------------------------------

            with curve_path.open(
                "w",
                newline="",
            ) as curve_file:

                curve_writer = csv.writer(
                    curve_file
                )

                curve_writer.writerow([
                    "model_type",
                    "size",
                    "train_seed",
                    "training_steps",
                    "mean_reward",
                    "std_reward",
                ])

                # =====================================================
                # STEP 0
                # Evaluate model BEFORE additional training
                # =====================================================

                initial_mean_reward, initial_std_reward = (
                    evaluate_fixed_seeds(
                        model,
                        env,
                        EVAL_SEEDS,
                    )
                )

                curve_writer.writerow([
                    "base_extended",
                    size,
                    seed,
                    0,
                    initial_mean_reward,
                    initial_std_reward,
                ])

                curve_file.flush()

                # The original model is initially the best model.
                best_mean_reward = initial_mean_reward
                best_checkpoint_path = load_path

                # Save a copy into policies_base so the
                # original model in policies/ is never overwritten.
                model.save(save_path)

                print()
                print(
                    "Initial performance before "
                    "additional training:"
                )

                print(
                    f"Mean reward: "
                    f"{initial_mean_reward:.2f} "
                    f"+/- {initial_std_reward:.2f}"
                )

                print(
                    "Initial model is currently "
                    "the best model."
                )

                # =====================================================
                # ADDITIONAL TRAINING
                # =====================================================

                for block in range(
                    1,
                    num_blocks + 1,
                ):

                    block_start_time = time.time()

                    print()
                    print("-" * 70)

                    print(
                        f"Starting block "
                        f"{block}/{num_blocks}"
                    )

                    print(
                        f"Additional training steps: "
                        f"{model.num_timesteps:,}"
                    )

                    print(
                        f"Best reward so far: "
                        f"{best_mean_reward:.2f}"
                    )

                    print("-" * 70)

                    # -------------------------------------------------
                    # Train
                    # -------------------------------------------------

                    model.learn(
                        total_timesteps=block_size,
                        log_interval=1,

                        # Reset counter at beginning because we want
                        # num_timesteps to represent ADDITIONAL steps.
                        reset_num_timesteps=(
                            block == 1
                        ),
                    )

                    # -------------------------------------------------
                    # Fixed-seed evaluation
                    # -------------------------------------------------

                    mean_reward, std_reward = (
                        evaluate_fixed_seeds(
                            model,
                            env,
                            EVAL_SEEDS,
                        )
                    )

                    # -------------------------------------------------
                    # Save learning curve point
                    # -------------------------------------------------

                    curve_writer.writerow([
                        "base_extended",
                        size,
                        seed,
                        model.num_timesteps,
                        mean_reward,
                        std_reward,
                    ])

                    curve_file.flush()

                    # -------------------------------------------------
                    # Timing
                    # -------------------------------------------------

                    block_elapsed = (
                        time.time()
                        - block_start_time
                    )

                    total_elapsed = (
                        time.time()
                        - start_time
                    )

                    avg_time_per_block = (
                        total_elapsed / block
                    )

                    remaining_blocks = (
                        num_blocks - block
                    )

                    estimated_remaining = (
                        avg_time_per_block
                        * remaining_blocks
                    )

                    # -------------------------------------------------
                    # Save checkpoint
                    # -------------------------------------------------

                    checkpoint_path = (
                        checkpoint_dir
                        / (
                            f"base_extended_block_"
                            f"{block:03d}_"
                            f"steps_{model.num_timesteps}_"
                            f"reward_{mean_reward:.2f}"
                        )
                    )

                    model.save(
                        checkpoint_path
                    )

                    # -------------------------------------------------
                    # Print block results
                    # -------------------------------------------------

                    print()

                    print(
                        f"Finished block "
                        f"{block}/{num_blocks}"
                    )

                    print(
                        f"Progress: "
                        f"{100 * block / num_blocks:.1f}%"
                    )

                    print(
                        f"Additional steps: "
                        f"{model.num_timesteps:,}"
                    )

                    print(
                        f"Mean reward: "
                        f"{mean_reward:.2f} "
                        f"+/- {std_reward:.2f}"
                    )

                    print(
                        f"Block time: "
                        f"{format_time(block_elapsed)}"
                    )

                    print(
                        f"Elapsed time: "
                        f"{format_time(total_elapsed)}"
                    )

                    print(
                        f"Estimated remaining: "
                        f"{format_time(estimated_remaining)}"
                    )

                    print(
                        f"Checkpoint saved: "
                        f"{checkpoint_path}.zip"
                    )

                    # -------------------------------------------------
                    # Best-model selection
                    # -------------------------------------------------

                    if mean_reward > best_mean_reward:

                        best_mean_reward = (
                            mean_reward
                        )

                        best_checkpoint_path = (
                            checkpoint_path
                        )

                        model.save(
                            save_path
                        )

                        print(
                            f"New best model saved to "
                            f"{save_path}.zip"
                        )

                    else:

                        print(
                            f"Best reward remains: "
                            f"{best_mean_reward:.2f}"
                        )

            # =========================================================
            # Finished this size / seed
            # =========================================================

            total_training_time = (
                time.time() - start_time
            )

            print()
            print("=" * 70)

            print(
                f"Training complete for "
                f"{size}x{size}, seed {seed}"
            )

            print(
                f"Best checkpoint: "
                f"{best_checkpoint_path}.zip"
            )

            print(
                f"Best mean reward: "
                f"{best_mean_reward:.2f}"
            )

            print(
                f"Total time: "
                f"{format_time(total_training_time)}"
            )

            print(
                f"Training curve saved to: "
                f"{curve_path}"
            )

            print("=" * 70)

            env.close()