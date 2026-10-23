from LunarLander.paths import APP_ROOT
import gymnasium as gym
from pathlib import Path
import numpy as np
import torch


INT_PATH = Path(
    f"{APP_ROOT}/models/qat_w4a4/qat_dqn_lander_256_256_seed_0_int4_weights.pt"
    
)

ENV_NAME = "LunarLander-v3"


class NumpyInt4QNetwork:
    def __init__(self, path):
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        self.layers = []

        for layer in checkpoint["layers"]:
            self.layers.append(
                {
                    "weight_int4": (
                        layer["weight_int4"]
                        .cpu()
                        .numpy()
                        .astype(np.int8)
                    ),
                    "scale": float(layer["scale"]),
                    "bias": (
                        layer["bias"]
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    ),
                }
            )

        self.activation_scales = [
            float(scale)
            for scale in checkpoint.get(
                "activation_scales",
                [],
            )
        ]

    def forward(self, obs):
        x = np.asarray(
            obs,
            dtype=np.float32,
        )

        if x.ndim == 1:
            x = x[None, :]

        activation_index = 0

        for layer_index, layer in enumerate(self.layers):
            weight_int4 = layer["weight_int4"]
            scale = layer["scale"]
            bias = layer["bias"]

            # INT4 values are physically stored as int8,
            # but are restricted to the logical INT4 range:
            #
            # -7 ... +7
            #
            # Convert to float for the current NumPy matmul,
            # then apply the quantization scale.
            x = (
                x @ weight_int4.T.astype(np.float32)
            ) * scale + bias

            if layer_index < len(self.layers) - 1:
                # ReLU
                x = np.maximum(x, 0.0)

                # 4-bit unsigned activation quantization
                # range = 0 ... 15
                if activation_index < len(self.activation_scales):
                    act_scale = self.activation_scales[
                        activation_index
                    ]

                    x_q = np.round(
                        x / act_scale
                    )

                    x_q = np.clip(
                        x_q,
                        0,
                        15,
                    )

                    x = x_q * act_scale

                    activation_index += 1

        return x

    def predict(self, obs):
        q_values = self.forward(obs)

        return int(
            np.argmax(
                q_values,
                axis=1,
            )[0]
        )


if __name__ == "__main__":
    env = gym.make(
        ENV_NAME,
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        render_mode="human",
    )

    model = NumpyInt4QNetwork(
        INT_PATH
    )

    for episode in range(5):
        obs, info = env.reset()

        total_reward = 0.0

        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = model.predict(obs)

            obs, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += reward

        print(
            f"Episode {episode}: "
            f"reward = {total_reward:.2f}"
        )

    env.close()