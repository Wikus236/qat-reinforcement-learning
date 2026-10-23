from LunarLander.paths import APP_ROOT
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import serial


ENV_NAME = "LunarLander-v3"
DEFAULT_PORT = "/dev/cu.usbserial-110"
DEFAULT_BAUD = 115200
DEFAULT_OUTPUT = Path(f"{APP_ROOT}/results/data/esp_benchmark_results.csv")

TEST_SEEDS = [
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

CSV_FIELDS = [
    "timestamp",
    "model_type",
    "size",
    "train_seed",
    "episodes",
    "total_steps",
    "device_inference_mean_ms",
    "device_inference_std_ms",
    "device_inference_min_ms",
    "device_inference_max_ms",
    "serial_roundtrip_mean_ms",
    "serial_roundtrip_std_ms",
    "serial_roundtrip_min_ms",
    "serial_roundtrip_max_ms",
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
    "test_seeds",
]


def make_env(render):
    return gym.make(
        ENV_NAME,
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        render_mode="human" if render else None,
    )


def wait_for_esp32(esp, timeout_seconds):
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        line = esp.readline().decode("utf-8", errors="ignore").strip()

        if not line:
            continue

        print(f"ESP32: {line}")

        if line == "READY":
            return

    print("No READY message received; continuing anyway.")


def request_model_info(esp):
    esp.write(b"i\n")

    while True:
        line = esp.readline().decode("utf-8", errors="ignore").strip()

        if not line:
            raise TimeoutError("Timed out waiting for ESP32 model information")

        if line.startswith("m,"):
            fields = line.split(",")

            if len(fields) != 3:
                raise RuntimeError(f"Invalid ESP32 model information: {line}")

            return fields[1], int(fields[2])

        if line.startswith("error"):
            raise RuntimeError(f"ESP32 returned: {line}")

        print(f"ESP32: {line}")


def request_action(esp, observation):
    observation = np.asarray(
        observation,
        dtype=np.float32,
    )
    message = (
        "s,"
        + ",".join(f"{value:.7g}" for value in observation)
        + "\n"
    )

    request_start = time.perf_counter()
    esp.write(message.encode("utf-8"))

    while True:
        line = esp.readline().decode("utf-8", errors="ignore").strip()

        if not line:
            raise TimeoutError("Timed out waiting for ESP32 action")

        if line.startswith("a,"):
            roundtrip_seconds = time.perf_counter() - request_start
            fields = line.split(",")

            if len(fields) < 3:
                raise RuntimeError(
                    "The ESP32 response does not include on-device inference "
                    "time. Upload the updated Arduino sketch before running "
                    "esp_benchmark.py."
                )

            action = int(fields[1])
            inference_microseconds = int(fields[2])

            return (
                action,
                inference_microseconds / 1000.0,
                roundtrip_seconds * 1000.0,
            )

        if line.startswith("error"):
            raise RuntimeError(f"ESP32 returned: {line}")

        print(f"ESP32: {line}")


def run_episode(env, esp, seed, step_delay):
    observation, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0
    total_steps = 0
    final_reward = 0.0
    device_times_ms = []
    roundtrip_times_ms = []

    while not terminated and not truncated:
        action, device_ms, roundtrip_ms = request_action(
            esp,
            observation,
        )
        device_times_ms.append(device_ms)
        roundtrip_times_ms.append(roundtrip_ms)

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1
        final_reward = reward

        if step_delay > 0:
            time.sleep(step_delay)

    return {
        "reward": total_reward,
        "steps": total_steps,
        "final_reward": final_reward,
        "terminated": terminated,
        "device_times_ms": device_times_ms,
        "roundtrip_times_ms": roundtrip_times_ms,
    }


def calculate_results(episodes):
    rewards = np.asarray(
        [episode["reward"] for episode in episodes],
        dtype=np.float64,
    )
    steps = np.asarray(
        [episode["steps"] for episode in episodes],
        dtype=np.float64,
    )
    final_rewards = np.asarray(
        [episode["final_reward"] for episode in episodes],
        dtype=np.float64,
    )
    terminated = np.asarray(
        [episode["terminated"] for episode in episodes],
        dtype=bool,
    )
    device_times_ms = np.asarray(
        [
            timing
            for episode in episodes
            for timing in episode["device_times_ms"]
        ],
        dtype=np.float64,
    )
    roundtrip_times_ms = np.asarray(
        [
            timing
            for episode in episodes
            for timing in episode["roundtrip_times_ms"]
        ],
        dtype=np.float64,
    )

    successful_landings = (
        terminated
        & np.isclose(final_rewards, 100.0)
    )
    failures = (
        terminated
        & np.isclose(final_rewards, -100.0)
    )

    return {
        "total_steps": int(steps.sum()),
        "device_inference_mean_ms": float(device_times_ms.mean()),
        "device_inference_std_ms": float(device_times_ms.std()),
        "device_inference_min_ms": float(device_times_ms.min()),
        "device_inference_max_ms": float(device_times_ms.max()),
        "serial_roundtrip_mean_ms": float(roundtrip_times_ms.mean()),
        "serial_roundtrip_std_ms": float(roundtrip_times_ms.std()),
        "serial_roundtrip_min_ms": float(roundtrip_times_ms.min()),
        "serial_roundtrip_max_ms": float(roundtrip_times_ms.max()),
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "median_reward": float(np.median(rewards)),
        "min_reward": float(rewards.min()),
        "max_reward": float(rewards.max()),
        "mean_steps": float(steps.mean()),
        "std_steps": float(steps.std()),
        "solve_rate": float(np.mean(rewards >= 200.0)),
        "success_rate": float(np.mean(successful_landings)),
        "failure_rate": float(np.mean(failures)),
    }


def append_results(output_path, row):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    write_header = (
        not output_path.exists()
        or output_path.stat().st_size == 0
    )

    with output_path.open("a", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=CSV_FIELDS,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a LunarLander model running on an ESP32 and append "
            "the results to a CSV file."
        )
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["int8", "float32"],
        help="Type of model currently uploaded to the ESP32",
    )
    parser.add_argument(
        "--size",
        required=True,
        type=int,
        choices=[16, 32, 64, 128, 256],
        help="Hidden-layer size currently uploaded to the ESP32",
    )
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--step-delay", type=float, default=0.0)
    parser.add_argument("--serial-timeout", type=float, default=2.0)
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    selected_seeds = [
        TEST_SEEDS[index % len(TEST_SEEDS)]
        for index in range(args.episodes)
    ]
    environment = make_env(render=args.render)

    print(
        f"Benchmarking {args.model_type} {args.size}x{args.size} "
        f"for {args.episodes} episodes"
    )
    print(f"Opening ESP32 on {args.port} at {args.baud} baud")

    try:
        with serial.Serial(
            args.port,
            args.baud,
            timeout=args.serial_timeout,
        ) as esp:
            time.sleep(2.0)
            esp.reset_input_buffer()
            wait_for_esp32(
                esp,
                timeout_seconds=3.0,
            )

            device_model_type, device_size = request_model_info(esp)

            if (
                device_model_type != args.model_type
                or device_size != args.size
            ):
                raise RuntimeError(
                    "Uploaded model does not match the benchmark labels: "
                    f"ESP32 reports {device_model_type} {device_size}x{device_size}, "
                    f"but the command specified {args.model_type} "
                    f"{args.size}x{args.size}."
                )

            print(
                f"ESP32 confirms {device_model_type} "
                f"{device_size}x{device_size}"
            )

            episode_results = []

            for episode_index, seed in enumerate(selected_seeds, start=1):
                result = run_episode(
                    environment,
                    esp,
                    seed,
                    args.step_delay,
                )
                episode_results.append(result)

                print(
                    f"Episode {episode_index}/{args.episodes}: "
                    f"seed={seed}, reward={result['reward']:.2f}, "
                    f"steps={result['steps']}, "
                    f"device_inference="
                    f"{np.mean(result['device_times_ms']):.3f} ms"
                )

        metrics = calculate_results(episode_results)
        row = {
            "timestamp": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "model_type": args.model_type,
            "size": args.size,
            "train_seed": args.train_seed,
            "episodes": args.episodes,
            **metrics,
            "test_seeds": ";".join(str(seed) for seed in selected_seeds),
        }
        append_results(
            args.output,
            row,
        )

        print()
        print(
            f"Reward: {metrics['mean_reward']:.2f} +/- "
            f"{metrics['std_reward']:.2f}"
        )
        print(
            "On-device inference: "
            f"{metrics['device_inference_mean_ms']:.3f} +/- "
            f"{metrics['device_inference_std_ms']:.3f} ms"
        )
        print(
            "Serial round trip: "
            f"{metrics['serial_roundtrip_mean_ms']:.3f} +/- "
            f"{metrics['serial_roundtrip_std_ms']:.3f} ms"
        )
        print(f"Appended results to {args.output}")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
