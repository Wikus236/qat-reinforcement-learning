from LunarLander.paths import APP_ROOT
import gymnasium as gym
import csv
import time
import numpy as np
from pathlib import Path
from rl_models.dqn import DQN


ENV_NAME = "LunarLander-v3"
MODEL_SIZES = [16, 32, 64, 128, 256]
SEEDS = [0, 1, 2, 3, 4]
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
            save_path = Path(f"{APP_ROOT}/models/initial_fp32/my_dqn_lander_{size}_{size}_seed_{seed}")
            save_path.parent.mkdir(exist_ok=True)
            checkpoint_dir = Path(
                f"{APP_ROOT}/models/initial_fp32/checkpoints/"
                f"{size}_{size}_seed_{seed}"
            )
            curve_path = Path(
                f"{APP_ROOT}/models/initial_fp32/training_curves/"
                f"base_{size}_{size}_seed_{seed}.csv"
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            curve_path.parent.mkdir(parents=True, exist_ok=True)

            env = gym.make(
                ENV_NAME,
                continuous=False,
                gravity=-10.0,
                enable_wind=False,
            )
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            policy_kwargs = dict(net_arch=[size, size])

            model = DQN(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=1e-4,
                buffer_size=100_000,
                learning_starts=10_000,
                batch_size=64,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1_000,
                exploration_fraction=0.2,
                exploration_final_eps=0.05,
                seed=seed,
                verbose=1,
            )
            print(f"Created new base model {size}x{size} with seed {seed}")

            block_size = 25_000
            total_timesteps = 2_500_000
            num_blocks = total_timesteps // block_size
            start_time = time.time()

            print()
            print("=" * 70)
            print(
                f"Starting base training for "
                f"{size}x{size}, seed {seed}"
            )
            print(f"Total timesteps: {total_timesteps:,}")
            print(f"Block size: {block_size:,}")
            print(f"Total blocks: {num_blocks}")
            print(f"Best model save path: {save_path}.zip")
            print(f"Curve CSV: {curve_path}")
            print("=" * 70)

            with curve_path.open("w", newline="") as curve_file:
                curve_writer = csv.writer(curve_file)

                curve_writer.writerow([
                    "model_type",
                    "size",
                    "train_seed",
                    "training_steps",
                    "mean_reward",
                    "std_reward",
                ])

                initial_mean_reward, initial_std_reward = evaluate_fixed_seeds(
                    model,
                    env,
                    EVAL_SEEDS,
                )

                curve_writer.writerow([
                    "base",
                    size,
                    seed,
                    model.num_timesteps,
                    initial_mean_reward,
                    initial_std_reward,
                ])
                curve_file.flush()

                best_mean_reward = initial_mean_reward
                model.save(save_path)

                print()
                print("Initial base model performance:")
                print(
                    f"Mean reward: {initial_mean_reward:.2f} "
                    f"+/- {initial_std_reward:.2f}"
                )

                for block in range(1, num_blocks + 1):
                    block_start_time = time.time()

                    print()
                    print("-" * 70)
                    print(f"Starting block {block}/{num_blocks}")
                    print(f"Training steps so far: {model.num_timesteps:,}")
                    print(f"Best reward so far: {best_mean_reward:.2f}")
                    print("-" * 70)

                    model.learn(
                        total_timesteps=block_size,
                        log_interval=1,
                        reset_num_timesteps=False,
                    )

                    mean_reward, std_reward = evaluate_fixed_seeds(
                        model,
                        env,
                        EVAL_SEEDS,
                    )

                    curve_writer.writerow([
                        "base",
                        size,
                        seed,
                        model.num_timesteps,
                        mean_reward,
                        std_reward,
                    ])
                    curve_file.flush()

                    checkpoint_path = (
                        checkpoint_dir
                        / (
                            f"base_block_"
                            f"{block:03d}_"
                            f"steps_{model.num_timesteps}_"
                            f"reward_{mean_reward:.2f}"
                        )
                    )
                    model.save(checkpoint_path)

                    block_elapsed = time.time() - block_start_time
                    total_elapsed = time.time() - start_time
                    avg_time_per_block = total_elapsed / block
                    remaining_blocks = num_blocks - block
                    estimated_remaining = avg_time_per_block * remaining_blocks

                    print()
                    print(f"Finished block {block}/{num_blocks}")
                    print(f"Progress: {100 * block / num_blocks:.1f}%")
                    print(f"Training steps: {model.num_timesteps:,}")
                    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                    print(f"Block time: {format_time(block_elapsed)}")
                    print(f"Elapsed time: {format_time(total_elapsed)}")
                    print(f"Estimated remaining: {format_time(estimated_remaining)}")
                    print(f"Checkpoint saved: {checkpoint_path}.zip")

                    if mean_reward > best_mean_reward:
                        best_mean_reward = mean_reward
                        model.save(save_path)
                        print(f"New best model saved to {save_path}.zip")
                    else:
                        print(f"Best reward remains: {best_mean_reward:.2f}")

            env.close()
