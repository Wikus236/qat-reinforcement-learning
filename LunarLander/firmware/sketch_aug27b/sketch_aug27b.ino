#include "model_data.h"

constexpr unsigned long BAUD_RATE = 115200;
constexpr int MAX_NEURONS = 256;

float current_values[MAX_NEURONS];
float next_values[MAX_NEURONS];

bool parseStateLine(String line, float state[MODEL_INPUT_SIZE]) {
  line.trim();

  if (!line.startsWith("s,")) {
    return false;
  }

  line.remove(0, 2);

  for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
    int comma_index = line.indexOf(',');
    String token;

    if (comma_index >= 0) {
      token = line.substring(0, comma_index);
      line.remove(0, comma_index + 1);
    } else {
      token = line;
      line = "";
    }

    token.trim();
    if (token.length() == 0) {
      return false;
    }

    state[i] = token.toFloat();
  }

  return true;
}

void runLayer(const float* input, float* output, int layer_index) {
  const int input_size = LAYER_INPUT_SIZES[layer_index];
  const int output_size = LAYER_OUTPUT_SIZES[layer_index];
  const model_weight_t* weights = LAYER_WEIGHTS[layer_index];
  const float* bias = LAYER_BIASES[layer_index];
  const float scale = LAYER_SCALES[layer_index];

  for (int out_i = 0; out_i < output_size; out_i++) {
    float acc = bias[out_i];

    for (int in_i = 0; in_i < input_size; in_i++) {
      int weight_index = out_i * input_size + in_i;
      float weight = float(weights[weight_index]);

      if (MODEL_WEIGHTS_ARE_QUANTIZED) {
        weight *= scale;
      }

      acc += input[in_i] * weight;
    }

    output[out_i] = acc;
  }
}

void applyReluAndActivationQuant(float* values, int size, int activation_index) {
  for (int i = 0; i < size; i++) {
    if (values[i] < 0.0f) {
      values[i] = 0.0f;
    }

    if (activation_index < MODEL_NUM_ACTIVATION_SCALES) {
      float act_scale = ACTIVATION_SCALES[activation_index];
      if (act_scale > 0.0f) {
        int q = int(roundf(values[i] / act_scale));
        q = constrain(q, 0, 255);
        values[i] = float(q) * act_scale;
      }
    }
  }
}

int predictAction(const float state[MODEL_INPUT_SIZE]) {
  for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
    current_values[i] = state[i];
  }

  int current_size = MODEL_INPUT_SIZE;
  int activation_index = 0;

  for (int layer_index = 0; layer_index < MODEL_NUM_LAYERS; layer_index++) {
    int output_size = LAYER_OUTPUT_SIZES[layer_index];
    runLayer(current_values, next_values, layer_index);

    if (layer_index < MODEL_NUM_LAYERS - 1) {
      applyReluAndActivationQuant(next_values, output_size, activation_index);
      activation_index++;
    }

    for (int i = 0; i < output_size; i++) {
      current_values[i] = next_values[i];
    }
    current_size = output_size;
  }

  int best_action = 0;
  float best_value = current_values[0];

  for (int i = 1; i < current_size; i++) {
    if (current_values[i] > best_value) {
      best_value = current_values[i];
      best_action = i;
    }
  }

  return best_action;
}

void setup() {
  Serial.begin(BAUD_RATE);
  delay(1000);
  Serial.println("READY");
}

void loop() {
  if (!Serial.available()) {
    return;
  }

  String line = Serial.readStringUntil('\n');
  line.trim();

  if (line == "i") {
    Serial.print("m,");
    Serial.print(MODEL_WEIGHTS_ARE_QUANTIZED ? "int8" : "float32");
    Serial.print(",");
    Serial.println(LAYER_OUTPUT_SIZES[0]);
    return;
  }

  float state[MODEL_INPUT_SIZE];

  if (!parseStateLine(line, state)) {
    Serial.println("error,bad_state_line");
    return;
  }

  unsigned long inference_start = micros();
  int action = predictAction(state);
  unsigned long inference_microseconds = micros() - inference_start;

  Serial.print("a,");
  Serial.print(action);
  Serial.print(",");
  Serial.println(inference_microseconds);
}
