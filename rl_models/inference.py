"""Deployment-style NumPy inference for FP32 and exported quantized DQNs.

PyTorch is used only to deserialize the saved checkpoints. Timed inference is
performed by these lightweight layer-by-layer NumPy implementations so all
model formats have the same prediction-call structure.
"""

from pathlib import Path
import re

import numpy as np
import torch


_LAYER_INDEX_PATTERN = re.compile(r"net\.(\d+)")


def _layer_index(name):
    match = _LAYER_INDEX_PATTERN.search(name)
    if match is None:
        raise ValueError(f"Cannot determine layer index from {name!r}")
    return int(match.group(1))


def _numpy_float32(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=np.float32)


def _numpy_integer(value, dtype):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=dtype)


def _scalar_float32(value):
    if hasattr(value, "item"):
        value = value.item()
    return np.float32(value)


class _NumpyQNetwork:
    def _prepare_input(self, observation):
        values = np.asarray(observation, dtype=np.float32)
        single_observation = values.ndim == 1

        if single_observation:
            values = values.reshape(1, -1)

        if values.ndim != 2:
            raise ValueError(
                "Expected one observation or a batch of observations, "
                f"received shape {values.shape}"
            )

        return values, single_observation

    def predict(self, observation, deterministic=True):
        del deterministic

        q_values = self.forward(observation)
        actions = np.argmax(q_values, axis=1)

        if actions.size == 1:
            return int(actions[0]), None

        return actions, None


class Float32QNetwork(_NumpyQNetwork):
    """FP32 checkpoint executed without the training-model/PyTorch wrapper."""

    def __init__(self, path):
        checkpoint = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint["q_net_state_dict"]

        weight_entries = [
            (name, value)
            for name, value in state_dict.items()
            if name.endswith(".weight") and value.ndim == 2
        ]
        weight_entries.sort(key=lambda item: _layer_index(item[0]))

        if not weight_entries:
            raise ValueError(f"No FP32 linear layers found in {path}")

        self.layers = []
        for weight_name, weight in weight_entries:
            bias_name = weight_name.removesuffix(".weight") + ".bias"
            if bias_name not in state_dict:
                raise KeyError(f"Missing bias tensor for {weight_name}")

            self.layers.append({
                "weight_float": _numpy_float32(weight),
                "bias": _numpy_float32(state_dict[bias_name]),
            })

        self.activation_scales = []
        self.hidden_sizes = tuple(
            layer["weight_float"].shape[0]
            for layer in self.layers[:-1]
        )

    def forward(self, observation):
        values, _ = self._prepare_input(observation)

        for layer_index, layer in enumerate(self.layers):
            values = values @ layer["weight_float"].T + layer["bias"]

            if layer_index < len(self.layers) - 1:
                values = np.maximum(values, np.float32(0.0))

        return values


class _QuantizedQNetwork(_NumpyQNetwork):
    weight_key = ""
    weight_dtype = None
    activation_qmax = 0

    def __init__(self, path):
        checkpoint = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=False,
        )

        self.layers = []
        for layer in checkpoint["layers"]:
            self.layers.append({
                self.weight_key: _numpy_integer(
                    layer[self.weight_key],
                    self.weight_dtype,
                ),
                "scale": _scalar_float32(layer["scale"]),
                "bias": _numpy_float32(layer["bias"]),
            })

        self.activation_scales = [
            _scalar_float32(scale)
            for scale in checkpoint.get("activation_scales", [])
        ]
        self.hidden_sizes = tuple(
            layer[self.weight_key].shape[0]
            for layer in self.layers[:-1]
        )

    def forward(self, observation):
        values, _ = self._prepare_input(observation)
        activation_index = 0

        for layer_index, layer in enumerate(self.layers):
            # Match the exported implementation: stored low-precision weights
            # are converted to FP32 and rescaled for each inference call.
            weight = (
                layer[self.weight_key].astype(np.float32)
                * layer["scale"]
            )
            values = values @ weight.T + layer["bias"]

            if layer_index < len(self.layers) - 1:
                values = np.maximum(values, np.float32(0.0))

                if activation_index < len(self.activation_scales):
                    scale = self.activation_scales[activation_index]
                    values = np.rint(values / scale)
                    values = np.clip(values, 0, self.activation_qmax)
                    values = values.astype(np.float32, copy=False) * scale
                    activation_index += 1

        return values


class Int8QNetwork(_QuantizedQNetwork):
    weight_key = "weight_int8"
    weight_dtype = np.int8
    activation_qmax = 255


class Int4QNetwork(_QuantizedQNetwork):
    # Logical INT4 values are physically stored in INT8 arrays.
    weight_key = "weight_int4"
    weight_dtype = np.int8
    activation_qmax = 15
