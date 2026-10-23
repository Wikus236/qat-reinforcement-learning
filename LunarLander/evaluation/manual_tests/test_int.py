from LunarLander.paths import APP_ROOT
import gymnasium as gym
from pathlib import Path
import numpy as np
import torch


INT_PATH = Path(f"{APP_ROOT}/models/qat_w8/qat_dqn_lander_16_16_seed_0_int8_weights.pt")
ENV_NAME = "LunarLander-v3"


class NumpyInt8QNetwork:
    def __init__(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        self.layers = []
        for layer in checkpoint["layers"]:
            self.layers.append(
                {
                    "weight_int8": layer["weight_int8"].cpu().numpy().astype(np.int8),
                    "scale": float(layer["scale"]),
                    "bias": layer["bias"].cpu().numpy().astype(np.float32),
                }
            )

        self.activation_scales = [
            float(scale)
            for scale in checkpoint.get("activation_scales", [])
        ]

    def forward(self, obs):
        x = np.asarray(obs, dtype=np.float32)

        if x.ndim == 1:
            x = x[None, :]

        activation_index = 0

        for layer_index, layer in enumerate(self.layers):
            weight_int8 = layer["weight_int8"]
            scale = layer["scale"]
            bias = layer["bias"]

            # Raw int8 matmul result:
            # x is float32, weight_int8 is int8.
            # Then multiply by scale to dequantize the dot product.
            x = (x @ weight_int8.T.astype(np.float32)) * scale + bias

            if layer_index < len(self.layers) - 1:
                x = np.maximum(x, 0.0)

                if activation_index < len(self.activation_scales):
                    act_scale = self.activation_scales[activation_index]
                    x_q = np.round(x / act_scale)
                    x_q = np.clip(x_q, 0, 255)
                    x = x_q * act_scale
                    activation_index += 1

        return x

    def predict(self, obs):
        q_values = self.forward(obs)
        return int(np.argmax(q_values, axis=1)[0])


if __name__ == "__main__":
    env = gym.make(
        ENV_NAME,
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        render_mode="human",
    )

    model = NumpyInt8QNetwork(INT_PATH)

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
        print(reward)
        print(total_reward)
        

    env.close()