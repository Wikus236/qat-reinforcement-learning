from LunarLander.paths import APP_ROOT
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from archive.lunar_lander.Lunar_Train_Continues import SAVE_PATH




ENV_NAME = "LunarLander-v3"
MODEL_PATH = SAVE_PATH



env = gym.make(ENV_NAME, continuous=True, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5,render_mode="human")

model = PPO.load(MODEL_PATH)
for episode in range(5):
    obs, info = env.reset()
    total_reward = 0.0

    terminated = False
    truncated = False

    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print(f"Episode {episode}: reward = {total_reward:.2f}")

env.close()