from LunarLander.paths import APP_ROOT
import gymnasium as gym
import csv
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from rl_models.qat_weights import DQN_QAT_W
from rl_models.qat import DQN_QAT
from rl_models.qat4 import DQN_QAT_4
from rl_models.qat_weights4 import DQN_QAT_W_4
from LunarLander.training.train_base_initial import MODEL_SIZES, SEEDS
from rl_models.inference import (
    Float32QNetwork,
    Int4QNetwork as NumpyInt4QNetwork,
    Int8QNetwork as NumpyInt8QNetwork,
)


POLICY_FOLDER = Path(f"{APP_ROOT}/models/extended_fp32")
QAT_W_POLICY_FOLDER = Path(f"{APP_ROOT}/models/qat_w8")
QAT_POLICY_FOLDER = Path(f"{APP_ROOT}/models/qat_w8a8")
QAT_4_POLICY_FOLDER = Path(f"{APP_ROOT}/models/qat_w4a4")
QAT_W_4_POLICY_FOLDER = Path(f"{APP_ROOT}/models/qat_w4")

ENV_NAME = "LunarLander-v3"

test_seeds = [
    42, 123, 456, 789, 111, 234, 23, 36, 964, 231,
    853, 131, 1331, 1234, 4321, 9876, 6789, 5432, 8765, 3456,
    102, 907, 275, 641, 819, 374, 590, 728, 163, 946,
    281, 734, 508, 692, 417, 860, 329, 574, 715, 983,
    1205, 2481, 3907, 4562, 5178, 6094, 7321, 8456, 9183, 10027,
    11482, 12639, 13704, 14891, 15936, 16275, 17420, 18563, 19648, 20791,
    21934, 23085, 24176, 25219, 26384, 27431, 28596, 29647, 30782, 31825,
    32970, 34013, 35158, 36201, 37346, 38489, 39524, 40667, 41712, 42855,
    43990, 45033, 46178, 47221, 48366, 49509, 50644, 51787, 52932, 54075,
    55210, 56353, 57498, 58641, 59786, 60929, 62064, 63207, 64352, 65495,
]


class Int8QNetwork:
    def __init__(self, path):
        self.checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.layers = self.checkpoint["layers"]
        self.activation_scales = self.checkpoint.get("activation_scales", [])

    def forward(self, obs):
        x = torch.tensor(obs, dtype=torch.float32)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        activation_index = 0

        for layer_index, layer in enumerate(self.layers):
            weight_int8 = layer["weight_int8"].float()
            scale = layer["scale"].float()
            bias = layer["bias"].float()

            weight = weight_int8 * scale
            x = F.linear(x, weight, bias)

            if layer_index < len(self.layers) - 1:
                x = F.relu(x)

                if activation_index < len(self.activation_scales):
                    act_scale = self.activation_scales[activation_index].float()
                    x_q = torch.round(x / act_scale).clamp(0, 255)
                    x = x_q * act_scale
                    activation_index += 1

        return x

    def predict(self, obs, deterministic=True):
        del deterministic

        with torch.no_grad():
            q_values = self.forward(obs)
            action = q_values.argmax(dim=1).item()

        return action, None

class Int4QNetwork:
    def __init__(self, path):
        self.checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.layers = self.checkpoint["layers"]
        self.activation_scales = self.checkpoint.get("activation_scales", [])

    def forward(self, obs):
        x = torch.tensor(obs, dtype=torch.float32)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        activation_index = 0

        for layer_index, layer in enumerate(self.layers):
            weight_int4 = layer["weight_int4"].float()
            scale = layer["scale"].float()
            bias = layer["bias"].float()

            weight = weight_int4 * scale
            x = F.linear(x, weight, bias)

            if layer_index < len(self.layers) - 1:
                x = F.relu(x)

                if activation_index < len(self.activation_scales):
                    act_scale = self.activation_scales[activation_index].float()

                    x_q = torch.round(x / act_scale).clamp(0, 15)
                    x = x_q * act_scale

                    activation_index += 1

        return x

    def predict(self, obs, deterministic=True):
        del deterministic

        with torch.no_grad():
            q_values = self.forward(obs)
            action = q_values.argmax(dim=1).item()

        return action, None


def evaluate_model(model, env):
    rewards = []
    steps = []
    final_rewards = []
    terminated_flags = []

    for seed in test_seeds:
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        total_steps = 0
        final_reward = 0.0

        terminated = False
        truncated = False

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            total_steps += 1
            final_reward = reward

        rewards.append(total_reward)
        steps.append(total_steps)
        final_rewards.append(final_reward)
        terminated_flags.append(terminated)

    rewards = np.array(rewards)
    steps = np.array(steps)
    final_rewards = np.array(final_rewards)
    terminated_flags = np.array(terminated_flags, dtype=bool)

    successful_landings = terminated_flags & np.isclose(final_rewards, 100.0)
    failures = terminated_flags & np.isclose(final_rewards, -100.0)

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "median_reward": np.median(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "mean_steps": np.mean(steps),
        "std_steps": np.std(steps),
        "solve_rate": np.mean(rewards >= 200),
        "success_rate": np.mean(successful_landings),
        "failure_rate": np.mean(failures),
    }


def build_models_to_test():
    models_to_test = []

    for size in MODEL_SIZES:
        for seed in SEEDS:
            model_specs = [
                (
                    "base_float",
                    POLICY_FOLDER / f"my_dqn_lander_{size}_{size}_seed_{seed}.zip",
                ),
                (
                    "qat_weight_only",
                    QAT_W_POLICY_FOLDER / f"qat_dqn_lander_{size}_{size}_seed_{seed}.zip",
                ),
                (
                    "int8_weight_only",
                    QAT_W_POLICY_FOLDER / f"qat_dqn_lander_{size}_{size}_seed_{seed}_int8_weights.pt",
                ),
                (
                    "qat_weight_activation",
                    QAT_POLICY_FOLDER / f"qat_dqn_lander_{size}_{size}_seed_{seed}.zip",
                ),
                (
                    "int8_weight_activation",
                    QAT_POLICY_FOLDER / f"qat_dqn_lander_{size}_{size}_seed_{seed}_int8_weights.pt",
                ),
                (
                    "qat_weight_activation_int4",
                    QAT_4_POLICY_FOLDER
                    / f"qat_dqn_lander_{size}_{size}_seed_{seed}.zip",
                ),
                (
                    "int4_weight_activation",
                    QAT_4_POLICY_FOLDER
                    / f"qat_dqn_lander_{size}_{size}_seed_{seed}_int4_weights.pt",
                ),
                (
                    "qat_weight_only_int4",
                    QAT_W_4_POLICY_FOLDER
                    / f"qat_dqn_lander_{size}_{size}_seed_{seed}.zip",
                ),
                (
                    "int4_weight_only",
                    QAT_W_4_POLICY_FOLDER
                    / f"qat_dqn_lander_{size}_{size}_seed_{seed}_int4_weights.pt",
                ),
            ]

            for model_type, model_path in model_specs:
                if model_path.exists():
                    models_to_test.append((model_type, size, seed, model_path))
                else:
                    print(f"Missing {model_type} {size}x{size} seed {seed}: {model_path}")

    return models_to_test


def load_model(model_type, model_path, env):
    if model_type == "base_float":
        return Float32QNetwork(model_path)

    if model_type == "qat_weight_only":
        return DQN_QAT_W.load(model_path, env=env)

    if model_type == "qat_weight_activation":
        return DQN_QAT.load(model_path, env=env)

    if model_type == "qat_weight_activation_int4":
        return DQN_QAT_4.load(model_path, env=env)
    if model_type == "qat_weight_only_int4":
        return DQN_QAT_W_4.load(model_path, env=env)

    if model_type in (
        "int8_weight_only",
        "int8_weight_activation",
    ):
        return NumpyInt8QNetwork(model_path)

    if model_type in (
        "int4_weight_only",
        "int4_weight_activation",
    ):
        return NumpyInt4QNetwork(model_path)

    raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    env = gym.make(
        ENV_NAME,
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
    )

    detailed_csv = f"{APP_ROOT}/results/data/model_results_detailed.csv"
    summary_csv = f"{APP_ROOT}/results/data/model_results_summary.csv"

    models_to_test = build_models_to_test()
    detailed_rows = []

    with open(detailed_csv, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "model_type",
            "size",
            "train_seed",
            "model",
            "mean_reward",
            "std_reward",
            "median_reward",
            "min_reward",
            "max_reward",
            "mean_steps",
            "std_steps",
            "solve_rate",
            "success_rate",
            "failure_rate",
        ])

        for model_type, size, seed, model_path in models_to_test:
            print(f"Testing {model_type}: {model_path.name}")

            model = load_model(model_type, model_path, env)
            results = evaluate_model(model, env)

            row = {
                "model_type": model_type,
                "size": size,
                "train_seed": seed,
                "model": model_path.name,
                **results,
            }
            detailed_rows.append(row)

            print(
                f"{model_path.name}: "
                f"{results['mean_reward']:.2f} +/- {results['std_reward']:.2f}"
            )

            writer.writerow([
                model_type,
                size,
                seed,
                model_path.name,
                results["mean_reward"],
                results["std_reward"],
                results["median_reward"],
                results["min_reward"],
                results["max_reward"],
                results["mean_steps"],
                results["std_steps"],
                results["solve_rate"],
                results["success_rate"],
                results["failure_rate"],
            ])

    with open(summary_csv, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "model_type",
            "size",
            "num_models",
            "mean_of_mean_rewards",
            "std_across_models",
            "best_mean_reward",
            "worst_mean_reward",
            "mean_solve_rate",
            "mean_success_rate",
            "mean_failure_rate",
            "mean_steps",
        ])

        for model_type in sorted(set(row["model_type"] for row in detailed_rows)):
            for size in MODEL_SIZES:
                rows = [
                    row for row in detailed_rows
                    if row["model_type"] == model_type and row["size"] == size
                ]

                if not rows:
                    continue

                mean_rewards = np.array([row["mean_reward"] for row in rows])
                solve_rates = np.array([row["solve_rate"] for row in rows])
                success_rates = np.array([row["success_rate"] for row in rows])
                failure_rates = np.array([row["failure_rate"] for row in rows])
                mean_steps = np.array([row["mean_steps"] for row in rows])

                writer.writerow([
                    model_type,
                    size,
                    len(rows),
                    np.mean(mean_rewards),
                    np.std(mean_rewards),
                    np.max(mean_rewards),
                    np.min(mean_rewards),
                    np.mean(solve_rates),
                    np.mean(success_rates),
                    np.mean(failure_rates),
                    np.mean(mean_steps),
                ])

    env.close()

    print(f"Saved detailed results to {detailed_csv}")
    print(f"Saved summary results to {summary_csv}")
