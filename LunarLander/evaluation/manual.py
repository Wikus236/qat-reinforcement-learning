from LunarLander.paths import APP_ROOT
import gymnasium as gym
import numpy as np
import pygame


ENV_NAME = "LunarLander-v3"

env = gym.make(
    ENV_NAME,
    continuous=True,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
    render_mode="human",
)

pygame.init()

for episode in range(5):
    obs, info = env.reset()
    total_reward = 0.0

    terminated = False
    truncated = False

    while not terminated and not truncated:
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        main_engine = 0.0
        side_engine = 0.0

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            main_engine = 1.0

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            side_engine = -1.0

        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            side_engine = 1.0

        if keys[pygame.K_ESCAPE]:
            terminated = True

        action = np.array([main_engine, side_engine], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print(f"Episode {episode}: reward = {total_reward:.2f}")

env.close()
pygame.quit()