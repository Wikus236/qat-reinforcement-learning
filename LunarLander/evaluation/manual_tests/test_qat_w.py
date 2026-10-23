from LunarLander.paths import APP_ROOT
import gymnasium as gym
from pathlib import Path
from rl_models.qat_weights import DQN_QAT_W


ENV_NAME = "LunarLander-v3"
# MODEL_PATH = Path(str(SAVE_PATH) + "_quantized")
# MODEL_PATH = Path(str(SAVE_PATH))  # Use the quantized model for testing
MODEL_PATH = Path(f"{APP_ROOT}/models/qat_w8/qat_dqn_lander_128_128")  # Use the quantized model for testing


env = gym.make(
    ENV_NAME,
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    render_mode="human",
)

model = DQN_QAT_W.load(MODEL_PATH, env=env)

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