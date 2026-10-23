from LunarLander.paths import APP_ROOT
import argparse
from pathlib import Path

import torch


DEFAULT_MODEL_PATH = Path(f"{APP_ROOT}/models/qat_w8/qat_dqn_lander_256_256_seed_0_int8_weights.pt")
DEFAULT_OUTPUT_PATH = Path(f"{APP_ROOT}/firmware/sketch_aug27b/model_data.h")


def scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def format_float(value):
    formatted = f"{float(scalar(value)):.9g}"

    if "." not in formatted and "e" not in formatted.lower():
        formatted += ".0"

    return formatted + "f"


def format_int8_array(values):
    flat = values.reshape(-1).tolist()
    lines = []
    for index in range(0, len(flat), 16):
        chunk = flat[index:index + 16]
        lines.append("    " + ", ".join(str(int(v)) for v in chunk))
    return ",\n".join(lines)


def format_float_array(values):
    flat = values.reshape(-1).tolist()
    lines = []
    for index in range(0, len(flat), 8):
        chunk = flat[index:index + 8]
        lines.append("    " + ", ".join(format_float(v) for v in chunk))
    return ",\n".join(lines)


def load_export_layers(checkpoint):
    if "layers" in checkpoint:
        return (
            checkpoint["layers"],
            checkpoint.get("activation_scales", []),
            True,
        )

    if "q_net_state_dict" in checkpoint:
        state_dict = checkpoint["q_net_state_dict"]
        layers = []

        for key, weight in state_dict.items():
            if not key.endswith(".weight") or weight.ndim != 2:
                continue

            bias_key = key.removesuffix(".weight") + ".bias"
            if bias_key not in state_dict:
                raise KeyError(f"Missing bias tensor for {key}")

            layers.append({
                "weight_float": weight,
                "bias": state_dict[bias_key],
            })

        if not layers:
            raise ValueError("No linear layers found in the FP32 checkpoint")

        return layers, [], False

    raise ValueError(
        "Unsupported checkpoint format. Expected quantized 'layers' "
        "or an FP32 'q_net_state_dict'."
    )


def export_header(model_path, output_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    layers, activation_scales, is_quantized = load_export_layers(checkpoint)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    weight_key = "weight_int8" if is_quantized else "weight_float"
    weight_type = "int8_t" if is_quantized else "float"

    input_size = int(layers[0][weight_key].shape[1])
    output_size = int(layers[-1][weight_key].shape[0])

    layer_inputs = []
    layer_outputs = []
    layer_scales = []
    text = []

    text.append("#pragma once")
    text.append("#include <stdint.h>")
    text.append("")
    text.append(f"using model_weight_t = {weight_type};")
    text.append(
        "constexpr bool MODEL_WEIGHTS_ARE_QUANTIZED = "
        f"{'true' if is_quantized else 'false'};"
    )
    text.append(f"constexpr int MODEL_INPUT_SIZE = {input_size};")
    text.append(f"constexpr int MODEL_OUTPUT_SIZE = {output_size};")
    text.append(f"constexpr int MODEL_NUM_LAYERS = {len(layers)};")
    text.append(f"constexpr int MODEL_NUM_ACTIVATION_SCALES = {len(activation_scales)};")
    text.append("")

    for layer_index, layer in enumerate(layers):
        if is_quantized:
            weight = layer[weight_key].cpu().numpy().astype("int8")
            formatted_weight = format_int8_array(weight)
            layer_scale = float(scalar(layer["scale"]))
        else:
            weight = layer[weight_key].cpu().numpy().astype("float32")
            formatted_weight = format_float_array(weight)
            layer_scale = 1.0

        bias = layer["bias"].cpu().numpy().astype("float32")
        out_size, in_size = weight.shape

        layer_inputs.append(in_size)
        layer_outputs.append(out_size)
        layer_scales.append(layer_scale)

        text.append(f"const model_weight_t LAYER_{layer_index}_WEIGHTS[] = {{")
        text.append(formatted_weight)
        text.append("};")
        text.append("")
        text.append(f"const float LAYER_{layer_index}_BIAS[] = {{")
        text.append(format_float_array(bias))
        text.append("};")
        text.append("")

    text.append("const int LAYER_INPUT_SIZES[] = {")
    text.append("    " + ", ".join(str(v) for v in layer_inputs))
    text.append("};")
    text.append("")
    text.append("const int LAYER_OUTPUT_SIZES[] = {")
    text.append("    " + ", ".join(str(v) for v in layer_outputs))
    text.append("};")
    text.append("")
    text.append("const float LAYER_SCALES[] = {")
    text.append("    " + ", ".join(format_float(v) for v in layer_scales))
    text.append("};")
    text.append("")

    text.append("const float ACTIVATION_SCALES[] = {")
    if activation_scales:
        text.append("    " + ", ".join(format_float(v) for v in activation_scales))
    text.append("};")
    text.append("")

    text.append("const model_weight_t* const LAYER_WEIGHTS[] = {")
    text.append("    " + ", ".join(f"LAYER_{i}_WEIGHTS" for i in range(len(layers))))
    text.append("};")
    text.append("")
    text.append("const float* const LAYER_BIASES[] = {")
    text.append("    " + ", ".join(f"LAYER_{i}_BIAS" for i in range(len(layers))))
    text.append("};")
    text.append("")

    output_path.write_text("\n".join(text))

    print(f"Exported {model_path} to {output_path}")
    print(f"Weight format: {'INT8 quantized' if is_quantized else 'FP32'}")
    print(f"Network: input={input_size}, output={output_size}, layers={layer_outputs}")


def main():
    parser = argparse.ArgumentParser(
        description="Export an INT8 QAT or FP32 baseline checkpoint to an Arduino header."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    export_header(args.model, args.output)


if __name__ == "__main__":
    main()
