from Pendulum.paths import APP_ROOT
import argparse
import sys
from pathlib import Path

import numpy as np


from rl_models.qat_weights import DQN_QAT_W
from Pendulum.environment.pendulum_env import LinearPendulumEnv
from Pendulum.training.train_pend_qat import (
    ACTION_VALUES,
    DOMAIN_RANDOMIZATION_RANGES,
    EPISODE_STEPS,
    PHASES,
    best_model_path,
    get_phase_config,
)


def make_env(randomized, evaluation_ranges, start_mode="swingup"):
    return LinearPendulumEnv(
        render_mode=None,
        max_episode_steps=EPISODE_STEPS,
        start_mode=start_mode,
        action_values=ACTION_VALUES,
        parameter_ranges=evaluation_ranges if randomized else None,
    )


def run_episode(model, env, seed):
    obs, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0
    steps = 0
    balanced_steps = 0
    streak = 0.0
    longest_streak = 0.0

    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        balanced_steps += int(info["balanced"])
        streak = streak + env.physics.dt if info["balanced"] else 0.0
        longest_streak = max(longest_streak, streak)

    return total_reward, steps, terminated, balanced_steps / steps, longest_streak


def evaluate(model, episodes, randomized, evaluation_ranges, start_mode="swingup"):
    env = make_env(
        randomized=randomized,
        evaluation_ranges=evaluation_ranges,
        start_mode=start_mode,
    )
    rewards = []
    steps = []
    rail_hits = 0
    balance_fractions = []
    longest_balances = []

    try:
        for episode in range(episodes):
            reward, episode_steps, terminated, balanced_fraction, longest_balance = run_episode(
                model,
                env,
                seed=10_000 + episode,
            )
            rewards.append(reward)
            steps.append(episode_steps)
            rail_hits += int(terminated)
            balance_fractions.append(balanced_fraction)
            longest_balances.append(longest_balance)
    finally:
        env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_steps": float(np.mean(steps)),
        "rail_hits": rail_hits,
        "episodes": episodes,
        "balance_fraction": float(np.mean(balance_fractions)),
        "longest_balance_seconds": float(np.mean(longest_balances)),
    }


def print_result(label, result):
    print(
        f"{label}: "
        f"reward={result['mean_reward']:.2f} +/- {result['std_reward']:.2f}, "
        f"min={result['min_reward']:.2f}, max={result['max_reward']:.2f}, "
        f"mean_steps={result['mean_steps']:.1f}, "
        f"balanced={result['balance_fraction']:.1%}, "
        f"mean_longest_balance={result['longest_balance_seconds']:.2f} s, "
        f"rail_hits={result['rail_hits']}/{result['episodes']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the pendulum policy on nominal and randomized physics."
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--phase",
        choices=PHASES.keys(),
        default="mild",
        help="Model phase to evaluate.",
    )
    args = parser.parse_args()

    config = get_phase_config(args.phase)
    model_path = best_model_path(config["model_path"])
    evaluation_ranges = config["parameter_ranges"] or DOMAIN_RANDOMIZATION_RANGES

    if not Path(str(model_path) + ".zip").exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.zip. "
            f"Run train_pend_qat.py --phase {args.phase} first."
        )

    model_env = make_env(randomized=False, evaluation_ranges=evaluation_ranges,
                         start_mode=config.get("eval_start_mode", config["start_mode"]))
    model = DQN_QAT_W.load(model_path, env=model_env)
    model_env.close()

    print(f"Evaluating phase: {args.phase} ({config['description']})")
    print_result(
        "Nominal physics",
        evaluate(
            model,
            args.episodes,
            randomized=False,
            evaluation_ranges=evaluation_ranges,
            start_mode=config.get("eval_start_mode", config["start_mode"]),
        ),
    )
    print_result(
        "Phase physics",
        evaluate(
            model,
            args.episodes,
            randomized=True,
            evaluation_ranges=evaluation_ranges,
            start_mode=config.get("eval_start_mode", config["start_mode"]),
        ),
    )


if __name__ == "__main__":
    main()
