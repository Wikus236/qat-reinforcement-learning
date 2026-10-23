from LunarLander.paths import APP_ROOT
import gymnasium as gym
from pathlib import Path
import torch
import torch.nn.functional as F

INT_PATH = Path(f"{APP_ROOT}/models/qat_w8/qat_dqn_lander_128_128")  # Use the quantized model for testing


ENV_NAME = "LunarLander-v3"



class Int8QNetwork:
    def __init__(self, path):
        self.checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.layers = self.checkpoint["layers"]

    def forward(self, obs):
        x = torch.tensor(obs, dtype=torch.float32)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        for layer_index, layer in enumerate(self.layers):
            weight_int8 = layer["weight_int8"].float()
            scale = layer["scale"]
            bias = layer["bias"].float()

            weight = weight_int8 * scale
            x = F.linear(x, weight, bias)

            if layer_index < len(self.layers) - 1:
                x = F.relu(x)

        return x

    def predict(self, obs):
        with torch.no_grad():
            q_values = self.forward(obs)
            action = q_values.argmax(dim=1).item()

        return action


if __name__ == "__main__":
    env = gym.make(
        ENV_NAME,
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        render_mode="human",
    )

    model = Int8QNetwork(INT_PATH)

    for episode in range(5):
        obs, info = env.reset()
        total_reward = 0.0

        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = model.predict(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        print(f"Episode {episode}: reward = {total_reward:.2f}")

    env.close()