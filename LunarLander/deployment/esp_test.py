from LunarLander.paths import APP_ROOT
import argparse
import time

import gymnasium as gym
import numpy as np
import serial


ENV_NAME = "LunarLander-v3"
DEFAULT_PORT = "/dev/cu.usbserial-110"
DEFAULT_BAUD = 115200
TEST_SEEDS = [42, 123, 456, 789, 111]


def make_env(render):
    render_mode = "human" if render else None
    return gym.make(
        ENV_NAME,
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        render_mode=render_mode,
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


def request_action(esp, obs):
    obs = np.asarray(obs, dtype=np.float32)
    message = "s," + ",".join(f"{value:.7g}" for value in obs) + "\n"
    request_start = time.perf_counter()
    esp.write(message.encode("utf-8"))

    while True:
        line = esp.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            raise TimeoutError("Timed out waiting for ESP32 action")

        if line.startswith("a,"):
            response_time = time.perf_counter() - request_start
            fields = line.split(",")
            action = int(fields[1])
            return action, response_time

        if line.startswith("error"):
            raise RuntimeError(f"ESP32 returned: {line}")

        print(f"ESP32: {line}")


def run_episode(env, esp, seed, step_delay):
    obs, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0
    steps = 0
    response_times = []

    while not terminated and not truncated:
        action, response_time = request_action(esp, obs)
        response_times.append(response_time)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if step_delay > 0:
            time.sleep(step_delay)

    return total_reward, steps, response_times


def main():
    parser = argparse.ArgumentParser(
        description="Run LunarLander in Python while an ESP32 chooses the actions."
    )
    parser.add_argument("--port", default=DEFAULT_PORT, help="ESP32 serial port")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud rate")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Show the LunarLander window")
    parser.add_argument("--step-delay", type=float, default=0.0, help="Delay after each environment step")
    parser.add_argument("--serial-timeout", type=float, default=2.0, help="Serial read timeout in seconds")
    args = parser.parse_args()

    env = make_env(render=args.render)

    print(f"Opening ESP32 on {args.port} at {args.baud} baud")
    with serial.Serial(args.port, args.baud, timeout=args.serial_timeout) as esp:
        time.sleep(2.0)
        esp.reset_input_buffer()
        wait_for_esp32(esp, timeout_seconds=3.0)

        try:
            all_response_times = []

            for episode in range(args.episodes):
                seed = TEST_SEEDS[episode % len(TEST_SEEDS)]
                reward, steps, response_times = run_episode(env, esp, seed, args.step_delay)
                all_response_times.extend(response_times)
                avg_response_ms = float(np.mean(response_times) * 1000.0)
                print(
                    f"Episode {episode + 1}/{args.episodes}: "
                    f"seed={seed}, reward={reward:.2f}, steps={steps}, "
                    f"avg_esp_response={avg_response_ms:.3f} ms"
                )

            if all_response_times:
                print(
                    "Overall ESP32 response time: "
                    f"avg={np.mean(all_response_times) * 1000.0:.3f} ms, "
                    f"min={np.min(all_response_times) * 1000.0:.3f} ms, "
                    f"max={np.max(all_response_times) * 1000.0:.3f} ms, "
                    f"samples={len(all_response_times)}"
                )
        finally:
            env.close()


if __name__ == "__main__":
    main()
