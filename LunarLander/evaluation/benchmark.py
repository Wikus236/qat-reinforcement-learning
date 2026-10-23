from LunarLander.paths import APP_ROOT
import time
import os
import csv
import gc
import tracemalloc
from pathlib import Path

# Keep the custom NumPy timing runs single-threaded and comparable across
# network sizes. These must be set before NumPy is imported.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

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


ENV_NAME = "LunarLander-v3"

POLICY_FOLDER = Path(f"{APP_ROOT}/models/extended_fp32")
QAT_W_POLICY_FOLDER = Path(f"{APP_ROOT}/models/qat_w8")
QAT_POLICY_FOLDER = Path(f"{APP_ROOT}/models/qat_w8a8")
QAT_4_POLICY_FOLDER = Path(f"{APP_ROOT}/models/qat_w4a4")
QAT_W_4_POLICY_FOLDER = Path(f"{APP_ROOT}/models/qat_w4")

OUTPUT_CSV = Path(f"{APP_ROOT}/results/data/model_benchmark_results.csv")


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
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        self.layers = checkpoint["layers"]
        self.activation_scales = checkpoint.get(
            "activation_scales",
            [],
        )

    def forward(self, obs):
        x = torch.as_tensor(
            obs,
            dtype=torch.float32,
        )

        if x.ndim == 1:
            x = x.unsqueeze(0)

        activation_index = 0

        for layer_index, layer in enumerate(self.layers):
            weight_int8 = layer["weight_int8"].float()
            scale = layer["scale"].float()
            bias = layer["bias"].float()

            # NOTE:
            # This dequantizes the INT8 weights back to FP32.
            # It is NOT native integer inference.
            weight = weight_int8 * scale

            x = F.linear(
                x,
                weight,
                bias,
            )

            if layer_index < len(self.layers) - 1:
                x = F.relu(x)

                if activation_index < len(self.activation_scales):
                    act_scale = (
                        self.activation_scales[activation_index]
                        .float()
                    )

                    x_q = torch.round(
                        x / act_scale
                    ).clamp(
                        0,
                        255,
                    )

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
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        self.layers = checkpoint["layers"]
        self.activation_scales = checkpoint.get(
            "activation_scales",
            [],
        )

    def forward(self, obs):
        x = torch.as_tensor(
            obs,
            dtype=torch.float32,
        )

        if x.ndim == 1:
            x = x.unsqueeze(0)

        activation_index = 0

        for layer_index, layer in enumerate(self.layers):
            weight_int4 = layer["weight_int4"].float()
            scale = layer["scale"].float()
            bias = layer["bias"].float()

            # Logical INT4 values are stored in an INT8 tensor.
            # This dequantizes them back to FP32 for F.linear.
            weight = weight_int4 * scale

            x = F.linear(
                x,
                weight,
                bias,
            )

            if layer_index < len(self.layers) - 1:
                x = F.relu(x)

                if activation_index < len(self.activation_scales):
                    act_scale = (
                        self.activation_scales[activation_index]
                        .float()
                    )

                    x_q = torch.round(
                        x / act_scale
                    ).clamp(
                        0,
                        15,
                    )

                    x = x_q * act_scale

                    activation_index += 1

        return x

    def predict(self, obs, deterministic=True):
        del deterministic

        with torch.no_grad():
            q_values = self.forward(obs)
            action = q_values.argmax(dim=1).item()

        return action, None


def build_models_to_test():
    models_to_test = []

    for size in MODEL_SIZES:
        for seed in SEEDS:
            model_specs = [
                (
                    "base_float",
                    POLICY_FOLDER
                    / f"my_dqn_lander_{size}_{size}_seed_{seed}.zip",
                ),
                (
                    "qat_weight_only",
                    QAT_W_POLICY_FOLDER
                    / f"qat_dqn_lander_{size}_{size}_seed_{seed}.zip",
                ),
                (
                    "int8_weight_only",
                    QAT_W_POLICY_FOLDER
                    / f"qat_dqn_lander_{size}_{size}_seed_{seed}_int8_weights.pt",
                ),
                (
                    "qat_weight_activation",
                    QAT_POLICY_FOLDER
                    / f"qat_dqn_lander_{size}_{size}_seed_{seed}.zip",
                ),
                (
                    "int8_weight_activation",
                    QAT_POLICY_FOLDER
                    / f"qat_dqn_lander_{size}_{size}_seed_{seed}_int8_weights.pt",
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
                    models_to_test.append(
                        (
                            model_type,
                            size,
                            seed,
                            model_path,
                        )
                    )
                else:
                    print(
                        f"Missing {model_type} "
                        f"{size}x{size} seed {seed}: "
                        f"{model_path}"
                    )

    return models_to_test


def load_model(
    model_type,
    model_path,
    env,
):
    if model_type == "base_float":
        return Float32QNetwork(model_path)

    if model_type == "qat_weight_only":
        return DQN_QAT_W.load(
            model_path,
            env=env,
        )

    if model_type == "qat_weight_activation":
        return DQN_QAT.load(
            model_path,
            env=env,
        )

    if model_type == "qat_weight_activation_int4":
        return DQN_QAT_4.load(
            model_path,
            env=env,
        )

    if model_type == "qat_weight_only_int4":
        return DQN_QAT_W_4.load(
            model_path,
            env=env,
        )

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

    raise ValueError(
        f"Unknown model type: {model_type}"
    )


def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


# ---------------------------------------------------------
# Actual parameter representation in the current Python files
# ---------------------------------------------------------

def estimate_model_params_mb(
    model,
    model_type,
):
    total_bytes = 0

    if model_type == "base_float":
        for layer in model.layers:
            total_bytes += layer["weight_float"].nbytes
            total_bytes += layer["bias"].nbytes

        return total_bytes / (1024 ** 2)

    if model_type in (
        "qat_weight_only",
        "qat_weight_only_int4",
        "qat_weight_activation",
        "qat_weight_activation_int4",
    ):
        for param in model.q_net.parameters():
            total_bytes += (
                param.numel()
                * param.element_size()
            )

        return total_bytes / (1024 ** 2)

    if model_type in (
        "int8_weight_only",
        "int8_weight_activation",
    ):
        for layer in model.layers:
            weight = layer["weight_int8"]
            scale = layer["scale"]
            bias = layer["bias"]

            total_bytes += weight.nbytes
            total_bytes += scale.nbytes
            total_bytes += bias.nbytes

        for scale in model.activation_scales:
            total_bytes += scale.nbytes

        return total_bytes / (1024 ** 2)

    if model_type in ("int4_weight_only","int4_weight_activation",):
        for layer in model.layers:
            weight = layer["weight_int4"]
            scale = layer["scale"]
            bias = layer["bias"]

            # IMPORTANT:
            # These are logical INT4 weights,
            # but physically stored in torch.int8.
            total_bytes += weight.nbytes
            total_bytes += scale.nbytes
            total_bytes += bias.nbytes

        for scale in model.activation_scales:
            total_bytes += scale.nbytes

        return total_bytes / (1024 ** 2)

    raise ValueError(
        f"Unknown model type: {model_type}"
    )


# ---------------------------------------------------------
# Theoretical deployment parameter storage
# ---------------------------------------------------------

def estimate_theoretical_params_mb(
    model,
    model_type,
):
    total_bytes = 0.0

    if model_type == "base_float":
        for layer in model.layers:
            total_bytes += layer["weight_float"].size * 4
            total_bytes += layer["bias"].size * 4

        return total_bytes / (1024 ** 2)

    if model_type in (
        "qat_weight_only",
        "qat_weight_only_int4",
        "qat_weight_activation",
        "qat_weight_activation_int4",
    ):
        for param in model.q_net.parameters():
            total_bytes += (
                param.numel()
                * param.element_size()
            )

    elif model_type in (
        "int8_weight_only",
        "int8_weight_activation",
    ):
        for layer in model.layers:
            # INT8 weight = 1 byte
            total_bytes += (
                layer["weight_int8"].size
            )

            # FP32 scale = 4 bytes
            total_bytes += 4

            # Bias remains FP32
            total_bytes += (
                layer["bias"].size * 4
            )

        # Activation scales remain FP32
        total_bytes += (
            len(model.activation_scales) * 4
        )

    elif model_type in (
        "int4_weight_only",
        "int4_weight_activation",
    ):
        for layer in model.layers:
            # Packed INT4:
            # two 4-bit weights per byte
            total_bytes += (
                layer["weight_int4"].size
                / 2.0
            )

            # FP32 scale
            total_bytes += 4

            # FP32 bias
            total_bytes += (
                layer["bias"].size * 4
            )

        total_bytes += (
            len(model.activation_scales) * 4
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    return total_bytes / (1024 ** 2)



# ---------------------------------------------------------
# Theoretical runtime activation storage
# ---------------------------------------------------------

def hidden_activation_elements(model):
    if hasattr(model, "hidden_sizes"):
        return sum(model.hidden_sizes)

    total_elements = 0

    for layer in model.layers[:-1]:
        if "weight_float" in layer:
            total_elements += layer["weight_float"].shape[0]
        elif "weight_int8" in layer:
            total_elements += layer["weight_int8"].shape[0]
        elif "weight_int4" in layer:
            total_elements += layer["weight_int4"].shape[0]

    return total_elements


def activation_bytes_per_element(model_type):
    if model_type in (
        "qat_weight_activation",
        "int8_weight_activation",
    ):
        return 1.0

    if model_type in (
        "qat_weight_activation_int4",
        "int4_weight_activation",
    ):
        # Theoretical packed 4-bit activations.
        # The current Python implementation still stores temporary tensors
        # in PyTorch tensor types, so this is not measured process memory.
        return 0.5

    return 4.0


def estimate_theoretical_activation_mb(
    model,
    model_type,
    batch_size=1,
):
    total_bytes = (
        hidden_activation_elements(model)
        * batch_size
        * activation_bytes_per_element(model_type)
    )

    return total_bytes / (1024 ** 2)


# ---------------------------------------------------------
# Inference timing
# ---------------------------------------------------------

def benchmark_inference_time(
    model,
    obs,
    num_runs=5_000,
    repeats=10,
):
    # Warm-up
    for _ in range(500):
        model.predict(
            obs,
            deterministic=True,
        )

    timings = []

    for _ in range(repeats):
        start = time.perf_counter()

        for _ in range(num_runs):
            model.predict(
                obs,
                deterministic=True,
            )

        end = time.perf_counter()

        avg_ms = (
            (end - start)
            / num_runs
            * 1000
        )

        timings.append(avg_ms)

    timings = np.asarray(
        timings,
        dtype=np.float64,
    )

    return {
        "mean_ms": float(timings.mean()),
        "std_ms": float(timings.std()),
        "min_ms": float(timings.min()),
        "max_ms": float(timings.max()),
    }


# ---------------------------------------------------------
# Python-side allocation measurement
#
# Supplemental metric only.
# Does not represent all PyTorch/native memory.
# ---------------------------------------------------------

def benchmark_peak_python_memory(
    model,
    obs,
    num_runs=1_000,
):
    gc.collect()

    tracemalloc.start()

    for _ in range(num_runs):
        model.predict(
            obs,
            deterministic=True,
        )

    _, peak = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    return peak / (1024 ** 2)


# ---------------------------------------------------------
# RL performance
# ---------------------------------------------------------

def evaluate_reward(
    model,
    env,
):
    rewards = []
    steps = []
    final_rewards = []
    terminated_flags = []

    for seed in test_seeds:
        obs, info = env.reset(
            seed=seed
        )

        total_reward = 0.0
        total_steps = 0
        final_reward = 0.0

        terminated = False
        truncated = False

        while (
            not terminated
            and not truncated
        ):
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward
            total_steps += 1
            final_reward = reward

        rewards.append(
            total_reward
        )

        steps.append(
            total_steps
        )

        final_rewards.append(
            final_reward
        )

        terminated_flags.append(
            terminated
        )

    rewards = np.asarray(
        rewards,
        dtype=np.float64,
    )

    steps = np.asarray(
        steps,
        dtype=np.float64,
    )

    final_rewards = np.asarray(
        final_rewards,
        dtype=np.float64,
    )

    terminated_flags = np.asarray(
        terminated_flags,
        dtype=bool,
    )

    successful_landings = (
        terminated_flags
        & np.isclose(final_rewards, 100.0)
    )

    failures = (
        terminated_flags
        & np.isclose(final_rewards, -100.0)
    )

    return {
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "median_reward": float(np.median(rewards)),
        "min_reward": float(rewards.min()),
        "max_reward": float(rewards.max()),
        "mean_steps": float(steps.mean()),
        "std_steps": float(steps.std()),
        "solve_rate": float(
            np.mean(rewards >= 200)
        ),
        "success_rate": float(
            np.mean(successful_landings)
        ),
        "failure_rate": float(
            np.mean(failures)
        ),
    }


if __name__ == "__main__":
    # Optional:
    # Makes CPU timing somewhat more repeatable for these small networks.
    torch.set_num_threads(1)

    env = gym.make(
        ENV_NAME,
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
    )

    benchmark_obs, info = env.reset(
        seed=42
    )

    # Make an independent copy so later env.reset/step calls
    # cannot affect the benchmark input.
    benchmark_obs = np.array(
        benchmark_obs,
        dtype=np.float32,
        copy=True,
    )

    models_to_test = build_models_to_test()

    with OUTPUT_CSV.open(
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "model_type",
            "size",
            "train_seed",
            "model",
            "inference_backend",

            "file_size_mb",

            "parameter_memory_mb",
            "theoretical_parameter_memory_mb",
            "theoretical_activation_memory_mb",
            "theoretical_runtime_memory_mb",

            "inference_mean_ms",
            "inference_std_ms",
            "inference_min_ms",
            "inference_max_ms",

            "peak_python_memory_mb",

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

        for (
            model_type,
            size,
            seed,
            model_path,
        ) in models_to_test:

            print("\n" + "=" * 70)

            print(
                f"Testing {model_type} | "
                f"{size}x{size} | "
                f"seed={seed}"
            )

            print("=" * 70)

            model = load_model(
                model_type,
                model_path,
                env,
            )

            file_size_mb = get_file_size_mb(
                model_path
            )

            parameter_memory_mb = (
                estimate_model_params_mb(
                    model,
                    model_type,
                )
            )

            theoretical_params_mb = (
                estimate_theoretical_params_mb(
                    model,
                    model_type,
                )
            )

            theoretical_activation_mb = (
                estimate_theoretical_activation_mb(
                    model,
                    model_type,
                )
            )

            theoretical_runtime_mb = (
                theoretical_params_mb
                + theoretical_activation_mb
            )

            timing_results = (
                benchmark_inference_time(
                    model,
                    benchmark_obs,
                )
            )

            peak_memory_mb = (
                benchmark_peak_python_memory(
                    model,
                    benchmark_obs,
                )
            )

            reward_results = evaluate_reward(
                model,
                env,
            )

            print(
                f"File size:              "
                f"{file_size_mb:.6f} MB"
            )

            print(
                f"Actual parameter memory:"
                f"{parameter_memory_mb:.6f} MB"
            )

            print(
                f"Theoretical deploy mem: "
                f"{theoretical_params_mb:.6f} MB"
            )

            print(
                f"Theoretical act. mem:   "
                f"{theoretical_activation_mb:.6f} MB"
            )

            print(
                f"Theoretical runtime mem:"
                f"{theoretical_runtime_mb:.6f} MB"
            )

            print(
                f"Inference time:         "
                f"{timing_results['mean_ms']:.6f} "
                f"+/- "
                f"{timing_results['std_ms']:.6f} ms"
            )

            print(
                f"Timing range:           "
                f"{timing_results['min_ms']:.6f} - "
                f"{timing_results['max_ms']:.6f} ms"
            )

            print(
                f"Peak Python memory:     "
                f"{peak_memory_mb:.6f} MB"
            )

            print(
                f"Reward:                 "
                f"{reward_results['mean_reward']:.2f} "
                f"+/- "
                f"{reward_results['std_reward']:.2f}"
            )

            writer.writerow([
                model_type,
                size,
                seed,
                model_path.name,
                (
                    "custom_numpy"
                    if model_type == "base_float"
                    or model_type.startswith("int8_")
                    or model_type.startswith("int4_")
                    else "pytorch_training_model"
                ),

                file_size_mb,

                parameter_memory_mb,
                theoretical_params_mb,
                theoretical_activation_mb,
                theoretical_runtime_mb,

                timing_results["mean_ms"],
                timing_results["std_ms"],
                timing_results["min_ms"],
                timing_results["max_ms"],

                peak_memory_mb,

                reward_results["mean_reward"],
                reward_results["std_reward"],
                reward_results["median_reward"],
                reward_results["min_reward"],
                reward_results["max_reward"],

                reward_results["mean_steps"],
                reward_results["std_steps"],

                reward_results["solve_rate"],
                reward_results["success_rate"],
                reward_results["failure_rate"],
            ])

            # Save after every model so a long benchmark
            # does not lose completed results.
            file.flush()

            del model
            gc.collect()

    env.close()

    print("\n" + "=" * 70)

    print(
        f"Saved benchmark results to "
        f"{OUTPUT_CSV.resolve()}"
    )

    print("=" * 70)
