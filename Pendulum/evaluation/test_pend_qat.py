from Pendulum.paths import APP_ROOT
import argparse
import sys
import time
from pathlib import Path


from rl_models.qat_weights import DQN_QAT_W
from Pendulum.environment.pendulum_env import LinearPendulumEnv
from Pendulum.training.train_pend_qat import (
    ACTION_VALUES,
    EPISODE_STEPS,
    NOMINAL_PENDULUM_LENGTH,
    PHASES,
    best_model_path,
    get_phase_config,
)


STEP_DELAY = 0.01
RESET_DELAY = 0.5


def make_env(start_mode="swingup"):
    return LinearPendulumEnv(
        render_mode="human",
        max_episode_steps=EPISODE_STEPS,
        start_mode=start_mode,
        action_values=ACTION_VALUES,
        parameter_ranges={
            "pendulum_length": (
                NOMINAL_PENDULUM_LENGTH,
                NOMINAL_PENDULUM_LENGTH,
            ),
        },
    )


def watch_policy(model, env):
    episode = 1

    while True:
        obs, info = env.reset()
        terminated = False
        truncated = False

        print(f"Starting episode {episode}. Press Ctrl+C to stop.")

        while not terminated and not truncated:
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            if STEP_DELAY > 0:
                time.sleep(STEP_DELAY)

        print("Episode ended. Resetting...\n")

        if RESET_DELAY > 0:
            time.sleep(RESET_DELAY)

        episode += 1


def main():
    parser = argparse.ArgumentParser(description="Watch a trained pendulum policy.")
    parser.add_argument(
        "--phase",
        choices=PHASES.keys(),
        default="mild",
        help="Model phase to test.",
    )
    args = parser.parse_args()

    model_path = best_model_path(get_phase_config(args.phase)["model_path"])

    if not Path(str(model_path) + ".zip").exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.zip. "
            f"Run train_pend_qat.py --phase {args.phase} first."
        )

    config = get_phase_config(args.phase)
    env = make_env(config.get("eval_start_mode", config["start_mode"]))
    params = env.physics.parameters
    print(
        "Testing physical model: "
        f"rail={2.0 * params.rail_limit:.3f} m, "
        f"pendulum={params.pendulum_length:.3f} m, "
        f"cart speed={params.max_cart_velocity:.3f} m/s, "
        f"cart acceleration={params.max_cart_acceleration:.3f} m/s^2"
    )

    model = DQN_QAT_W.load(
        model_path,
        env=env,
    )

    try:
        watch_policy(
            model,
            env,
        )
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
