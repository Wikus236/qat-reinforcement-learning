from Pendulum.paths import APP_ROOT
import math
from dataclasses import dataclass, replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class PendulumParameters:
    # Measured effective length of the weighted pendulum on the 800 mm rail.
    pendulum_length: float = 0.2476
    gravity: float = 9.81
    # Four measurements with consistent bottom calibration:
    # mean effective length 0.2476 m; mean damping 0.03870.
    pendulum_damping: float = 0.039
    rail_limit: float = 0.4
    # Velocity comes from the measured rail travel and pulse rate. Acceleration
    # is fitted to the observed response of the physical mechanism.
    max_cart_velocity: float = 0.309
    max_cart_acceleration: float = 2.464
    # Arduino uses 10% of the full rail as the margin at each end.
    rail_safety_margin_fraction: float = 0.1


class LinearPendulumPhysics:
    def __init__(self, parameters=None, dt=0.01, hardware_safety=True):
        self.parameters = parameters or PendulumParameters()
        self.dt = dt
        self.hardware_safety = hardware_safety
        self.safety_override_direction = 0
        self.state = np.zeros(4, dtype=np.float64)

    def reset(self, x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0):
        self.safety_override_direction = 0
        self.state = np.array(
            [x, x_dot, theta, theta_dot],
            dtype=np.float64,
        )
        return self.state.copy()

    def step(self, motor_command):
        params = self.parameters
        dt = self.dt
        x, x_dot, theta, theta_dot = self.state

        command = float(np.clip(motor_command, -1.0, 1.0))
        if self.hardware_safety:
            command = self._apply_rail_safety(command, x, x_dot)
        target_velocity = command * params.max_cart_velocity
        max_velocity_change = params.max_cart_acceleration * dt

        velocity_change = float(
            np.clip(
                target_velocity - x_dot,
                -max_velocity_change,
                max_velocity_change,
            )
        )

        new_x_dot = x_dot + velocity_change
        x_ddot = (new_x_dot - x_dot) / dt

        x_dot = new_x_dot
        x += x_dot * dt

        if x >= params.rail_limit:
            x = params.rail_limit
            x_dot = 0.0
            x_ddot = 0.0
        elif x <= -params.rail_limit:
            x = -params.rail_limit
            x_dot = 0.0
            x_ddot = 0.0

        theta_ddot = (
            params.gravity * np.sin(theta)
            - x_ddot * np.cos(theta)
            - params.pendulum_damping * theta_dot
        ) / params.pendulum_length

        theta_dot += theta_ddot * dt
        theta += theta_dot * dt
        theta = self.wrap_angle(theta)

        self.state = np.array(
            [x, x_dot, theta, theta_dot],
            dtype=np.float64,
        )

        return self.state.copy()

    def _apply_rail_safety(self, command, x, velocity):
        params = self.parameters
        margin = 2.0 * params.rail_limit * params.rail_safety_margin_fraction
        braking_distance = velocity * velocity / (2.0 * params.max_cart_acceleration)
        reversal_position = max(0.0, params.rail_limit - margin - braking_distance)
        release_position = params.rail_limit - 2.0 * margin

        # Match updateDqnAction(): hold the inward command until the cart has
        # left the safety zone, then allow the policy to choose again.
        if self.safety_override_direction < 0:
            if x <= release_position:
                self.safety_override_direction = 0
            else:
                return -1.0
        elif self.safety_override_direction > 0:
            if x >= -release_position:
                self.safety_override_direction = 0
            else:
                return 1.0

        if velocity > 0.0 and x >= reversal_position:
            self.safety_override_direction = -1
            return -1.0
        if velocity < 0.0 and x <= -reversal_position:
            self.safety_override_direction = 1
            return 1.0
        return command

    def is_at_rail_limit(self):
        return abs(self.state[0]) >= self.parameters.rail_limit

    @staticmethod
    def wrap_angle(angle):
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class LinearPendulumEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 60,
    }

    def __init__(
        self,
        render_mode=None,
        max_episode_steps=800,
        start_mode="swingup",
        action_values=None,
        dt=0.01,
        parameter_ranges=None,
        swingup_reward=False,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.start_mode = start_mode
        self.swingup_reward = swingup_reward
        self.nominal_parameters = PendulumParameters()
        self.parameter_ranges = parameter_ranges or {}
        self.nominal_dt = dt
        self.physics = LinearPendulumPhysics(
            parameters=replace(self.nominal_parameters),
            dt=dt,
        )
        self.step_count = 0

        if action_values is None:
            action_values = [-1.0, 1.0]

        self.action_values = np.asarray(
            action_values,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self.action_values))

        limit = self._observation_limit("rail_limit")
        max_velocity = self._observation_limit("max_cart_velocity")
        self.observation_space = spaces.Box(
            low=np.array([-limit, -max_velocity, -np.pi, -np.inf], dtype=np.float32),
            high=np.array([limit, max_velocity, np.pi, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None

        if self.render_mode == "human":
            self._open_renderer()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self._randomize_physics_parameters()

        if self.start_mode == "upright":
            theta = self.np_random.uniform(-0.1, 0.1)
        elif self.start_mode == "down" or (
            self.start_mode == "mixed_down" and self.np_random.random() < 0.8
        ):
            theta = self.physics.wrap_angle(
                np.pi + self.np_random.uniform(-0.1, 0.1)
            )
        else:
            theta = self.np_random.uniform(-np.pi, np.pi)

        state = self.physics.reset(
            x=options.get("x", 0.0),
            x_dot=options.get("x_dot", 0.0),
            theta=options.get("theta", theta),
            theta_dot=options.get("theta_dot", self.np_random.uniform(-0.1, 0.1)),
        )
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(state), {}

    def _randomize_physics_parameters(self):
        parameters = replace(self.nominal_parameters)

        for name, bounds in self.parameter_ranges.items():
            low, high = bounds
            value = float(self.np_random.uniform(low, high))

            if name == "dt":
                self.physics.dt = value
            elif hasattr(parameters, name):
                setattr(parameters, name, value)
            else:
                raise ValueError(f"Unknown pendulum parameter: {name}")

        if "dt" not in self.parameter_ranges:
            self.physics.dt = self.nominal_dt

        self.physics.parameters = parameters

    def _observation_limit(self, parameter_name):
        if parameter_name in self.parameter_ranges:
            return max(abs(value) for value in self.parameter_ranges[parameter_name])

        return abs(getattr(self.nominal_parameters, parameter_name))

    def step(self, action):
        motor_command = float(self.action_values[int(action)])
        state = self.physics.step(motor_command)
        self.step_count += 1

        reward = self._reward(state, motor_command)

        terminated = self.physics.is_at_rail_limit()
        truncated = self.step_count >= self.max_episode_steps

        if terminated:
            reward -= 100.0

        if self.render_mode == "human":
            self.render(reward=reward, motor_command=motor_command)

        info = {
            "balanced": abs(state[2]) <= math.radians(15.0) and abs(state[3]) <= 1.0,
            "safety_override": self.physics.safety_override_direction != 0,
        }
        return self._get_obs(state), reward, terminated, truncated, info

    def _reward(self, state, motor_command):
        x, x_dot, theta, theta_dot = state

        del x_dot

        rail_limit = self.physics.parameters.rail_limit
        normalized_position = min(abs(x) / rail_limit, 1.0)

        upright_reward = 0.30 * (math.cos(theta) + 1.0) / 2.0
        # A stationary upright pendulum earns more than a fast upright crossing.
        balance_reward = 0.75 * math.exp(
            -(theta / math.radians(15.0)) ** 2 - (theta_dot / 1.0) ** 2
        )
        # Unlike tanh, this keeps distinguishing fast rotations from slow ones.
        angular_velocity_penalty = -0.002 * theta_dot ** 2
        if self.swingup_reward:
            # Retain full braking pressure near upright, but allow swing-up
            # speed elsewhere. Keep 10% of the original rotation penalty.
            upright_proximity = math.exp(-(theta / math.radians(45.0)) ** 2)
            angular_velocity_penalty *= 0.1 + 0.9 * upright_proximity
        barrier_penalty = -0.20 * (normalized_position ** 4)
        action_penalty = -0.01 * abs(motor_command)

        return float(
            upright_reward
            + balance_reward
            + angular_velocity_penalty
            + barrier_penalty
            + action_penalty
        )

    def _get_obs(self, state):
        return np.asarray(state, dtype=np.float32)

    def render(self, reward=None, motor_command=0.0):
        if self.render_mode != "human":
            return None

        if self._screen is None:
            self._open_renderer()

        pygame = self._pygame
        screen = self._screen
        clock = self._clock
        font = self._font

        x, x_dot, theta, theta_dot = self.physics.state
        params = self.physics.parameters

        width, height = 1000, 600
        rail_y = int(height * 0.65)
        rail_margin = 100
        rail_pixels = width - 2 * rail_margin
        pixels_per_metre = rail_pixels / (2 * params.rail_limit)

        def world_x_to_screen_x(value):
            return int(width // 2 + value * pixels_per_metre)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()

        screen.fill((245, 245, 245))

        rail_left = rail_margin
        rail_right = width - rail_margin
        pygame.draw.line(screen, (40, 40, 40), (rail_left, rail_y), (rail_right, rail_y), 6)

        cart_x = world_x_to_screen_x(x)
        cart_width = 50
        cart_height = 25
        cart_top = rail_y - cart_height
        cart_rect = pygame.Rect(cart_x - cart_width // 2, cart_top, cart_width, cart_height)
        pygame.draw.rect(screen, (60, 110, 200), cart_rect, border_radius=8)

        pivot_x = cart_x
        pivot_y = cart_top
        pendulum_length_px = params.pendulum_length * pixels_per_metre
        mass_x = pivot_x + pendulum_length_px * math.sin(theta)
        mass_y = pivot_y - pendulum_length_px * math.cos(theta)

        pygame.draw.line(screen, (180, 50, 50), (pivot_x, pivot_y), (int(mass_x), int(mass_y)), 8)
        pygame.draw.circle(screen, (30, 30, 30), (int(mass_x), int(mass_y)), 16)
        pygame.draw.circle(screen, (40, 40, 40), (pivot_x, pivot_y), 8)

        pygame.draw.rect(screen, (40, 40, 40), (rail_left - 4, rail_y - 80, 8, 80))
        pygame.draw.rect(screen, (40, 40, 40), (rail_right - 4, rail_y - 80, 8, 80))

        lines = [
            f"x: {x:.3f} m",
            f"x_dot: {x_dot:.3f} m/s",
            f"theta: {math.degrees(theta):.2f} deg",
            f"theta_dot: {theta_dot:.3f} rad/s",
            f"motor: {motor_command:.1f}",
        ]

        if reward is not None:
            lines.append(f"reward: {reward:.2f}")

        for index, line in enumerate(lines):
            surface = font.render(line, True, (40, 40, 40))
            screen.blit(surface, (20, 20 + index * 30))

        pygame.display.flip()
        clock.tick(self.metadata["render_fps"])

        return None

    def _open_renderer(self):
        import pygame

        self._pygame = pygame
        pygame.init()
        self._screen = pygame.display.set_mode((1000, 600))
        pygame.display.set_caption("DQN Linear Pendulum")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont(None, 28)

    def close(self):
        if self._pygame is not None:
            self._pygame.quit()
        self._screen = None
        self._clock = None
        self._font = None
