from LunarLander.paths import APP_ROOT
from pathlib import Path
import torch


WEIGHTS_PATH = Path(f"{APP_ROOT}/models/qat_w8a8/qat_dqn_lander_16_16_int8_weights.pt")


checkpoint = torch.load(
    WEIGHTS_PATH,
    map_location="cpu",
    weights_only=False,
)

print(checkpoint.keys())

for layer_index, layer in enumerate(checkpoint["layers"]):
    weight_int8 = layer["weight_int8"]
    scale = layer["scale"]
    bias = layer["bias"]

    print(f"\nLayer {layer_index}")
    print(f"weight_int8 shape: {weight_int8.shape}")
    print(f"weight_int8 dtype: {weight_int8.dtype}")
    print(f"weight_int8 min: {weight_int8.min().item()}")
    print(f"weight_int8 max: {weight_int8.max().item()}")
    print(f"scale: {scale}")
    print(f"bias shape: {bias.shape}")
    print(f"bias dtype: {bias.dtype}")
    print(f"bias min: {bias.min().item()}")
    print(f"bias max: {bias.max().item()}")

    dequantized_weight = weight_int8.float() * scale

    print(f"dequantized min: {dequantized_weight.min().item()}")
    print(f"dequantized max: {dequantized_weight.max().item()}")
    print(f"dequantized mean: {dequantized_weight.mean().item()}")
    print(f"dequantized std: {dequantized_weight.std().item()}")