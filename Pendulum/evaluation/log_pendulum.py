"""Log one continuous 30-second simulation of the trained base policy.

Run: python Pendulum/log_pendulum.py
Time is simulated time. Angle is measured from upright (0 degrees); the
initial hanging-down position is -180 degrees in the CSV. Plots shift
hanging down to 0 degrees and wrap at +/-180 degrees. Rail contact is recorded,
but the physics continues with its existing rail clamps without resetting.
"""

from Pendulum.paths import APP_ROOT

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

HERE = APP_ROOT

from rl_models.qat_weights import DQN_QAT_W
from Pendulum.plotting.position_plot import position_figure
from Pendulum.environment.pendulum_env import LinearPendulumEnv
from Pendulum.training.train_pend_qat import ACTION_VALUES, BASE_MODEL_PATH, NOMINAL_PENDULUM_LENGTH, best_model_path


def log_run(model, env, duration=30.0, seed=0):
    dt = env.physics.dt
    steps = round(duration / dt)
    if duration <= 0 or not math.isclose(steps * dt, duration, abs_tol=1e-9):
        raise ValueError("Duration must be positive and a multiple of the simulation timestep.")
    obs, _ = env.reset(seed=seed, options={
        "x": 0.0, "x_dot": 0.0, "theta": -math.pi, "theta_dot": 0.0,
    })
    rows = []
    for step in range(steps + 1):
        x, x_dot, theta, theta_dot = env.physics.state
        length = env.physics.parameters.pendulum_length
        rows.append({
            "time_s": round(step * dt, 10),
            "cart_position_m": float(x),
            "cart_velocity_m_s": float(x_dot),
            "angle_rad": float(theta),
            "angle_deg": math.degrees(theta),
            "angular_velocity_rad_s": float(theta_dot),
            "tip_x_m": float(x + length * math.sin(theta)),
            "tip_y_m": float(length * math.cos(theta)),
            "at_rail_limit": int(env.physics.is_at_rail_limit()),
            "safety_override": int(getattr(env.physics, "safety_override_direction", 0)),
            "action_to_next_step": "",
            "motor_command_to_next_step": "",
        })
        if step == steps:
            break
        action, _ = model.predict(obs, deterministic=True)
        command = float(env.action_values[int(action)])
        rows[-1]["action_to_next_step"] = int(action)
        rows[-1]["motor_command_to_next_step"] = command
        # Use the same physics as env.step, without episode termination/reset.
        obs = np.asarray(env.physics.step(command), dtype=np.float32)
    return rows


def save_outputs(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pendulum_position.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    times = np.array([row["time_s"] for row in rows])
    angles = np.array([row["angle_deg"] for row in rows])
    fig = position_figure(times, angles, "Base-model pendulum position over time", wrap=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"pendulum_position.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=best_model_path(BASE_MODEL_PATH),
                        help="Trained base checkpoint path, with or without .zip.")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Duration in simulated seconds (default: 30).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "results/simulation")
    args = parser.parse_args()
    checkpoint = args.model if args.model.suffix == ".zip" else Path(str(args.model) + ".zip")
    if not checkpoint.is_file():
        parser.error(f"Trained model not found: {checkpoint}. Use --model to specify its location.")
    env = LinearPendulumEnv(start_mode="down", action_values=ACTION_VALUES,
                            parameter_ranges={"pendulum_length": (NOMINAL_PENDULUM_LENGTH, NOMINAL_PENDULUM_LENGTH)})
    try:
        model = DQN_QAT_W.load(checkpoint, env=env)
        if hasattr(model, "freeze_activation_scales"):
            model.freeze_activation_scales()
        rows = log_run(model, env, args.duration, args.seed)
        csv_path = save_outputs(rows, args.output_dir)
        print(f"Saved {len(rows)} samples over {rows[-1]['time_s']:g} simulated seconds.")
        print(f"CSV: {csv_path}")
        print(f"Graphs: {args.output_dir / 'pendulum_position.png'} and .pdf")
        contacts = sum(row["at_rail_limit"] for row in rows)
        print(f"Rail-limit samples: {contacts}. No episode resets were performed.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
