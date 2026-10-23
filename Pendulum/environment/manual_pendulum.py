from Pendulum.paths import APP_ROOT
import math
import time

import pygame

from Pendulum.environment.pendulum_env import LinearPendulumEnv


ACTION_VALUES = [-1.0, 0.0, 1.0]


def reset_at_bottom(env):
    observation, _ = env.reset(
        options={
            "x": 0.0,
            "x_dot": 0.0,
            "theta": math.pi,
            "theta_dot": 0.0,
        }
    )
    return observation


def selected_action():
    keys = pygame.key.get_pressed()
    left_pressed = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right_pressed = keys[pygame.K_RIGHT] or keys[pygame.K_d]

    if left_pressed == right_pressed:
        return 1
    if left_pressed:
        return 0
    return 2


def main():
    env = LinearPendulumEnv(
        render_mode="human",
        max_episode_steps=1_000_000,
        start_mode="down",
        action_values=ACTION_VALUES,
        dt=0.01,
    )
    env.metadata["render_fps"] = 100
    reset_at_bottom(env)
    pygame.display.set_caption("Manual Linear Pendulum")

    params = env.physics.parameters
    print("Manual pendulum control")
    print("Left/A = move left, Right/D = move right, release = stop")
    print("R = reset at bottom, Esc = quit")
    print(
        f"Rail: {2.0 * params.rail_limit:.3f} m | "
        f"pendulum: {params.pendulum_length:.3f} m | "
        f"speed: {params.max_cart_velocity:.3f} m/s | "
        f"acceleration: {params.max_cart_acceleration:.3f} m/s^2"
    )

    running = True
    try:
        while running:
            reset_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        reset_requested = True

            if not running:
                break

            if reset_requested:
                reset_at_bottom(env)
                continue

            action = selected_action()
            _, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                print("Rail limit reached. Resetting at the centre.")
                time.sleep(0.25)
                reset_at_bottom(env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
