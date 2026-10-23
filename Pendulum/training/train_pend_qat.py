from Pendulum.paths import APP_ROOT
import argparse
import csv
import math
import signal
from datetime import datetime, timezone
import sys
from pathlib import Path

import numpy as np


from rl_models.qat_weights import DQN_QAT_W
from Pendulum.training.mild_dqn import MildDQN
from Pendulum.environment.pendulum_env import LinearPendulumEnv
from Pendulum.training.base_training_guard import BaseTrainingGuard, EXPLORATION_FLOOR, RECOVERY_STEPS


MODEL_DIR = APP_ROOT / "models/original"
MODEL_NAME = "dqn_qat_linear_pendulum"
BASE_MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}_base"
MILD_RANDOM_MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}_mild_random"
WIDE_RANDOM_MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}_wide_random"
MODEL_PATH = MILD_RANDOM_MODEL_PATH
LOAD_EXISTING_MODEL = True
ACTION_VALUES = [-1.0, 0.0, 1.0]
EPISODE_STEPS = 2000
NOMINAL_PENDULUM_LENGTH = 0.2476
NOMINAL_PENDULUM_DAMPING = 0.039
# Refinement uses mostly the learned policy even with a new step counter.
MILD_EXPLORATION_INITIAL_EPS = 0.02
MILD_EXPLORATION_FINAL_EPS = 0.005
MILD_EXPLORATION_FRACTION = 0.20
BASE_TRAINING_STEPS = 50_000_000
BASE_EXPLORATION_DECAY_STEPS = 2_500_000
MILD_TRAINING_STEPS = 1_000_000
MILD_LEARNING_RATE = 1e-5

BASE_PARAMETER_RANGES = {
    "pendulum_length": (
        NOMINAL_PENDULUM_LENGTH,
        NOMINAL_PENDULUM_LENGTH,
    ),
    "pendulum_damping": (
        NOMINAL_PENDULUM_DAMPING,
        NOMINAL_PENDULUM_DAMPING,
    ),
}

DOMAIN_RANDOMIZATION_RANGES = {
    "pendulum_length": (0.23, 0.27),
    "pendulum_damping": (0.030, 0.050),
    "max_cart_velocity": (0.24, 0.38),
    "max_cart_acceleration": (1.8, 3.0),
    "rail_limit": (0.36, 0.44),
    "dt": (0.008, 0.012),
}
# Mild starts near the bottom with only small pendulum variations.
# Since T = 2*pi*sqrt(length/g), squaring the period factors gives the
# effective-length bounds for a +/-1% small-angle period variation.
MILD_PERIOD_VARIATION = 0.01
MILD_DAMPING_VARIATION = 0.05
ACTIVE_PARAMETER_RANGES = {
    "pendulum_length": (
        NOMINAL_PENDULUM_LENGTH * (1.0 - MILD_PERIOD_VARIATION) ** 2,
        NOMINAL_PENDULUM_LENGTH * (1.0 + MILD_PERIOD_VARIATION) ** 2,
    ),
    "pendulum_damping": (
        NOMINAL_PENDULUM_DAMPING * (1.0 - MILD_DAMPING_VARIATION),
        NOMINAL_PENDULUM_DAMPING * (1.0 + MILD_DAMPING_VARIATION),
    ),
}

PHASES = {
    "base": {
        "model_path": BASE_MODEL_PATH,
        "start_paths": [],
        "parameter_ranges": BASE_PARAMETER_RANGES,
        "start_mode": "swingup",
        "description": "fixed-physics base model with random-angle starts",
    },
    "mild": {
        "model_path": MILD_RANDOM_MODEL_PATH,
        "start_paths": [Path(str(BASE_MODEL_PATH) + "_best")],
        "restart_from_best": True,
        "parameter_ranges": ACTIVE_PARAMETER_RANGES,
        "start_mode": "swingup",
        "eval_start_mode": "swingup",
        "swingup_reward": False,
        "description": "random-angle refinement with light period and damping variation",
    },
    "wide": {
        "model_path": WIDE_RANDOM_MODEL_PATH,
        "start_paths": [MILD_RANDOM_MODEL_PATH, BASE_MODEL_PATH],
        "parameter_ranges": DOMAIN_RANDOMIZATION_RANGES,
        "start_mode": "swingup",
        "description": "wide domain-randomized model with random-angle starts",
    },
}


def get_phase_config(phase):
    return PHASES[phase]


def make_env(seed, parameter_ranges=ACTIVE_PARAMETER_RANGES, start_mode="swingup", swingup_reward=False):
    env = LinearPendulumEnv(
        render_mode=None,
        max_episode_steps=EPISODE_STEPS,
        start_mode=start_mode,
        action_values=ACTION_VALUES,
        parameter_ranges=parameter_ranges,
        swingup_reward=swingup_reward,
    )

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env


def load_or_create_model(env, config, resume=None):
    model_class = MildDQN if config.get("restart_from_best", False) else DQN_QAT_W
    restart_from_best = config.get("restart_from_best", False)
    if resume is None:
        resume = not restart_from_best
    model_path = config["model_path"]
    model_file = Path(str(model_path) + ".zip")

    if LOAD_EXISTING_MODEL and resume and model_file.exists():
        print(f"Loading existing model from {model_file}")
        return model_class.load(model_path, env=env), True

    for start_path in config["start_paths"]:
        start_file = Path(str(start_path) + ".zip")
        if LOAD_EXISTING_MODEL and start_file.exists():
            print(f"Starting {config['description']} from {start_file}")
            model = model_class.load(start_path, env=env)
            if restart_from_best:
                model.num_timesteps = 0
                print("Mild training starts from best base weights at step 0")
            return model, not restart_from_best

    if LOAD_EXISTING_MODEL and restart_from_best:
        raise FileNotFoundError(
            f"Best base checkpoint not found at {config['start_paths'][0]}.zip. "
            "Train the base phase first."
        )

    if LOAD_EXISTING_MODEL:
        checked_paths = [model_path] + config["start_paths"]
        checked_files = ", ".join(f"{path}.zip" for path in checked_paths)
        print(f"No existing model found at {checked_files}; starting a new model")
    else:
        print("Starting a new model")

    model = DQN_QAT_W(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            net_arch=[128, 128],
        ),
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=50_000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.5,
        exploration_final_eps=0.05,
        seed=42,
        verbose=1,
    )

    return model, False


def best_model_path(model_path):
    return Path(str(model_path) + "_best")


def evaluate_model(model, env, episode_count, seed):
    rewards = []
    lengths = []
    rail_failures = 0
    balance_fractions = []
    longest_balance_seconds = []
    capture_times = []

    for episode in range(episode_count):
        observation, info = env.reset(seed=seed + episode)
        total_reward = 0.0
        episode_length = 0
        balanced_steps = 0
        balance_streak = 0.0
        longest_streak = 0.0
        elapsed_seconds = 0.0
        capture_time = None

        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_length += 1
            balanced_steps += int(info["balanced"])
            balance_streak = balance_streak + env.physics.dt if info["balanced"] else 0.0
            longest_streak = max(longest_streak, balance_streak)
            elapsed_seconds += env.physics.dt
            if capture_time is None and balance_streak >= 1.0:
                capture_time = elapsed_seconds - balance_streak

            if terminated or truncated:
                rail_failures += int(terminated)
                break

        rewards.append(total_reward)
        lengths.append(episode_length)
        balance_fractions.append(balanced_steps / episode_length)
        longest_balance_seconds.append(longest_streak)
        if capture_time is not None:
            capture_times.append(capture_time)


    return {
        "mean_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "rail_failures": rail_failures,
        "rail_failure_rate": rail_failures / episode_count,
        "capture_rate": len(capture_times) / episode_count,
        "mean_capture_seconds": float(np.mean(capture_times)) if capture_times else None,
        "balance_fraction": float(np.mean(balance_fractions)),
        "longest_balance_seconds": float(np.mean(longest_balance_seconds)),
    }


def model_selection_score(metrics):
    # Balance comes first, then capture reliability/speed and overall reward.
    values = [metrics[key] for key in (
        "longest_balance_seconds", "balance_fraction", "capture_rate", "mean_reward"
    )]
    if not all(math.isfinite(value) for value in values):
        return (float("-inf"),) * 5
    capture_time = metrics["mean_capture_seconds"]
    return (
        metrics["longest_balance_seconds"],
        metrics["balance_fraction"],
        metrics["capture_rate"],
        -capture_time if capture_time is not None else float("-inf"),
        metrics["mean_reward"],
    )


def append_training_log(path, phase, model, metrics, event, total_steps):
    row = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase, "steps": model.num_timesteps, "event": event,
        **metrics, "learning_rate": model.learning_rate,
        "exploration_rate": model._exploration_rate(total_steps),
        "learning_starts": model.learning_starts,
        "recoveries": getattr(model, "training_guard_state", {}).get("recoveries", 0),
    }
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def training_plan(phase, timesteps=None):
    if phase == "both":
        if timesteps is not None:
            raise ValueError("For --phase both, use --base-steps and --mild-steps.")
        return [("base", BASE_TRAINING_STEPS), ("mild", MILD_TRAINING_STEPS)]
    defaults = {"base": BASE_TRAINING_STEPS, "mild": MILD_TRAINING_STEPS,
                "wide": 2_000_000}
    return [(phase, timesteps if timesteps is not None else defaults[phase])]


def main():
    parser = argparse.ArgumentParser(description="Train the 3-action QAT pendulum DQN.")
    parser.add_argument(
        "--phase", choices=[*PHASES, "both"], default="both",
        help="Run one phase, or base followed by mild.",
    )
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override the step count for a single phase.")
    parser.add_argument("--base-steps", type=int, default=BASE_TRAINING_STEPS)
    parser.add_argument("--mild-steps", type=int, default=MILD_TRAINING_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume the existing mild checkpoint instead of restarting from best base.",
    )
    args = parser.parse_args()
    try:
        plan = training_plan(args.phase, args.timesteps)
    except ValueError as error:
        parser.error(str(error))
    if args.phase == "both":
        plan = [("base", args.base_steps), ("mild", args.mild_steps)]
    if any(steps <= 0 for _, steps in plan) or args.eval_freq <= 0 or args.eval_episodes <= 0:
        parser.error("Step counts, evaluation frequency and episode count must be positive.")
    def terminate(signum, frame):
        raise KeyboardInterrupt("Termination requested")
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        for phase, steps in plan:
            train_phase(args, phase, steps)
    except KeyboardInterrupt:
        print("Training interrupted; the latest saved checkpoint can be resumed.")
        raise SystemExit(130)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def train_phase(args, phase, total_steps):
    config = get_phase_config(phase)
    
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = make_env(args.seed, config["parameter_ranges"], config["start_mode"],
                   config.get("swingup_reward", False))
    eval_env = make_env(args.seed + 10_000, config["parameter_ranges"],
                       config.get("eval_start_mode", config["start_mode"]),
                       config.get("swingup_reward", False))
    model, loading_existing_model = load_or_create_model(
        env, config, resume=args.resume if phase == "mild" else None,
    )

    if phase != "base":
        # Base recovery metadata must not alter mild/wide exploration schedules.
        model.training_guard_state = {}
    if phase == "base":
        model.exploration_final_eps = EXPLORATION_FLOOR
        # Keep the original 2.5M-step base annealing duration even on a 50M run.
        model.exploration_fraction = min(1.0, BASE_EXPLORATION_DECAY_STEPS / total_steps)
        if loading_existing_model:
            # Checkpoints omit replay data; refill before resuming weight updates.
            model.learning_starts = max(model.learning_starts, model.num_timesteps + RECOVERY_STEPS)

    if phase == "mild":
        if not loading_existing_model:
            model.learning_starts = 50_000
        model.exploration_initial_eps = MILD_EXPLORATION_INITIAL_EPS
        model.exploration_final_eps = MILD_EXPLORATION_FINAL_EPS
        model.exploration_fraction = MILD_EXPLORATION_FRACTION
        model.learning_rate = MILD_LEARNING_RATE
        # Loading restores the base optimizer too; update its actual rate.
        for group in model.optimizer.param_groups:
            group["lr"] = MILD_LEARNING_RATE
        if loading_existing_model:
            model.learning_starts = model.num_timesteps + 50_000
        teacher = DQN_QAT_W.load(config["start_paths"][0], env=env)
        # Use the best base's actual discount factor, even when resuming an
        # older mild checkpoint that used a different objective.
        model.gamma = teacher.gamma
        demo_env = make_env(args.seed + 30_000, config["parameter_ranges"], "swingup", False)
        print("Collecting successful base trajectories for mild refinement...")
        model.prepare_refinement(teacher, demo_env, args.seed + 30_000)
        print(f"Retained {len(model.demo_buffer):,} base transitions; each batch uses 25% base replay")

    print("Training DQN-QAT on custom linear pendulum environment")
    params = env.physics.parameters
    print(
        "Physical model: "
        f"rail={2.0 * params.rail_limit:.3f} m, "
        f"pendulum={params.pendulum_length:.3f} m, "
        f"cart speed={params.max_cart_velocity:.3f} m/s, "
        f"cart acceleration={params.max_cart_acceleration:.3f} m/s^2"
    )
    print(f"Phase: {phase} ({config['description']})")
    print("Actions: 0 = left, 1 = stop, 2 = right")
    print(
        f"Exploration: {model.exploration_initial_eps:.0%} to "
        f"{model.exploration_final_eps:.0%} random actions over "
        f"{model.exploration_fraction:.0%} of training"
    )
    print(f"Learning rate: {model.learning_rate:g}")
    print(f"Training this phase for {total_steps:,} additional steps")
    print(f"Episode length: {EPISODE_STEPS} steps")
    print(
        "Episodes start 80% near the bottom and 20% at random angles" if config["start_mode"] == "mixed_down"
        else "Episodes start near the bottom" if config["start_mode"] == "down"
        else "Episodes start at random angles around the full circle"
    )
    print("Hardware rail safety enabled; best model selected by sustained balance")
    if phase == "mild":
        print(f"Mild discount factor inherited from best base: {model.gamma:g}")
        print("Mild uses the same reward as base training")
        print("Mild training and evaluations start at random angles")
    print(f"Saving model to {config['model_path']}.zip")
    if phase == "base":
        print(
            "Fixed pendulum parameters: "
            f"length={NOMINAL_PENDULUM_LENGTH:.3f} m, "
            f"damping={NOMINAL_PENDULUM_DAMPING:.3f}"
        )
    elif config["parameter_ranges"]:
        print("Domain randomization enabled:")
        for name, bounds in config["parameter_ranges"].items():
            print(f"  {name}: {bounds[0]} to {bounds[1]}")
    else:
        print("Domain randomization disabled: training fixed-physics base model")

    selected_best_path = best_model_path(config["model_path"])
    selected_best_file = Path(str(selected_best_path) + ".zip")
    best_score = (float("-inf"),) * 5
    best_metrics = None
    guard = BaseTrainingGuard(model, selected_best_path, args.eval_freq) if phase == "base" else None
    log_path = MODEL_DIR / f"{MODEL_NAME}_{phase}_training.csv"
    last_evaluated_timestep = -1

    if selected_best_file.exists():
        existing_best = DQN_QAT_W.load(selected_best_path, env=eval_env)
        existing_metrics = evaluate_model(
            existing_best,
            eval_env,
            args.eval_episodes,
            args.seed + 20_000,
        )
        best_score = model_selection_score(existing_metrics)
        best_metrics = existing_metrics
        print(
            f"Existing best model: mean_reward={existing_metrics['mean_reward']:.2f} | "
            f"std={existing_metrics['reward_std']:.2f} | "
            f"mean_length={existing_metrics['mean_length']:.1f} | "
            f"balanced={existing_metrics['balance_fraction']:.1%} | "
            f"mean_longest_balance={existing_metrics['longest_balance_seconds']:.2f} s | "
            f"rail_failures={existing_metrics['rail_failures']}/"
            f"{args.eval_episodes}"
        )

    def evaluate_and_save(trained_model):
        nonlocal best_score, best_metrics, last_evaluated_timestep

        if trained_model.num_timesteps == last_evaluated_timestep:
            return

        metrics = evaluate_model(
            trained_model,
            eval_env,
            args.eval_episodes,
            args.seed + 20_000,
        )
        last_evaluated_timestep = trained_model.num_timesteps
        diagnostics = getattr(trained_model, "last_refinement_diagnostics", None)
        if diagnostics:
            print(f"Mild update: TD loss={diagnostics['td_loss']:.4f}; "
                  f"policy preservation loss={diagnostics['preservation_loss']:.4f}; "
                  f"base action agreement={diagnostics['teacher_agreement']:.1%}")

        capture_text = (
            f"{metrics['mean_capture_seconds']:.2f} s"
            if metrics["mean_capture_seconds"] is not None else "not captured"
        )
        print(
            f"Evaluation at {trained_model.num_timesteps} steps | "
            f"mean_reward={metrics['mean_reward']:.2f} | "
            f"capture_rate={metrics['capture_rate']:.1%} | "
            f"mean_capture_time={capture_text} | "
            f"std={metrics['reward_std']:.2f} | "
            f"mean_length={metrics['mean_length']:.1f} | "
            f"balanced={metrics['balance_fraction']:.1%} | "
            f"mean_longest_balance={metrics['longest_balance_seconds']:.2f} s | "
            f"rail_failures={metrics['rail_failures']}/{args.eval_episodes}"
        )

        improved = (
            (guard is None or guard.is_finite(trained_model, metrics))
            and (best_metrics is None or model_selection_score(metrics) > best_score)
        )
        if improved:
            best_score = model_selection_score(metrics)
            best_metrics = metrics.copy()
        event = guard.observe(trained_model, metrics, best_metrics, improved) if guard else (
            "new_best" if improved else "evaluated"
        )
        if improved:
            trained_model.save(selected_best_path)
            print(
                f"New best model saved to {selected_best_path}.zip "
                f"with mean longest balance {metrics['longest_balance_seconds']:.2f} s"
            )
        if event == "restored_best":
            print(
                f"Recovery: restored best weights; learning rate={trained_model.learning_rate:g}; "
                f"replay buffer cleared; updates resume after step {trained_model.learning_starts:,}"
            )
        elif event == "exploration_boost":
            print("Progress stalled: temporary exploration boost enabled")
        trained_model.save(config["model_path"])
        append_training_log(log_path, phase, trained_model, metrics, event, total_steps)

    if guard:
        print("Base safeguards: evaluate every "
              f"{args.eval_freq:,} steps; 10% exploration floor; rollback only after 3M steps "
              "and reliable balance; require 6 bad evaluations; 500K recovery cooldown")
        print(f"Active base learning rate: {model.learning_rate:g}")
        print(f"Evaluation history: {log_path}")
    try:
        # Record the starting policy before any updates, for both base and mild.
        evaluate_and_save(model)
        model.learn(
            total_timesteps=total_steps,
            log_interval=1,
            reset_num_timesteps=not loading_existing_model,
            callback=evaluate_and_save,
            callback_freq=args.eval_freq,
        )
        evaluate_and_save(model)
        model.save(config["model_path"])
        print(f"Saved trained model to {config['model_path']}.zip")
        print(f"Best model: {selected_best_path}.zip")
    except (KeyboardInterrupt, Exception):
        model.save(config["model_path"])
        print(f"Saved interruption checkpoint at step {model.num_timesteps:,}")
        raise
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
