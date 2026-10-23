.DEFAULT_GOAL := help
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
PYTHON ?= $(REPO_ROOT)/.venv/bin/python
export PYTHONPATH := $(REPO_ROOT)$(if $(PYTHONPATH),:$(PYTHONPATH))

lander_train_continues:
	$(PYTHON) -m archive.lunar_lander.Lunar_Train_Continues

lander_test_continues:
	$(PYTHON) -m archive.lunar_lander.Lunar_Test_Continues

lander_manual:
	$(PYTHON) -m LunarLander.evaluation.manual

train_lib:
	$(PYTHON) -m archive.lunar_lander.Lunar_Train

test_lib:
	$(PYTHON) -m archive.lunar_lander.Lunar_Test

base:
	$(PYTHON) -m LunarLander.training.train_base_initial

train_base:
	$(PYTHON) -m LunarLander.training.train_base_extended

test_base:
	$(PYTHON) -m LunarLander.evaluation.manual_tests.test_base

train_qat_w:
	$(PYTHON) -m LunarLander.training.train_qat_weights8

test_qat_w:
	$(PYTHON) -m LunarLander.evaluation.manual_tests.test_qat_w

train_qat:
	$(PYTHON) -m LunarLander.training.train_qat8

test_qat:
	$(PYTHON) -m LunarLander.evaluation.manual_tests.test_qat

test_int_4:
	$(PYTHON) -m LunarLander.evaluation.manual_tests.test_int_4

compare:
	$(PYTHON) -m LunarLander.evaluation.compare

test_int_w:
	$(PYTHON) -m LunarLander.evaluation.manual_tests.test_int_w

test_int:
	$(PYTHON) -m LunarLander.evaluation.manual_tests.test_int

train_qat_4:
	$(PYTHON) -m LunarLander.training.train_qat4

train_qat_4_w:
	$(PYTHON) -m LunarLander.training.train_qat_weights4

size:
	$(PYTHON) -m LunarLander.evaluation.size_compare

plot_compare:
	$(PYTHON) -m LunarLander.plotting.plot_compare

plot_benchmark:
	$(PYTHON) -m LunarLander.plotting.plot_benchmark

plot_steps:
	$(PYTHON) -m LunarLander.plotting.plot_steps

combine_plots:
	$(PYTHON) -m LunarLander.plotting.combine_plots

inspect:
	$(PYTHON) -m LunarLander.evaluation.inspect_weights

benchmark:
	$(PYTHON) -m LunarLander.evaluation.benchmark

table:
	$(PYTHON) -m LunarLander.reporting.generate_table_data

esp_test_16:
	$(PYTHON) -m LunarLander.deployment.export_esp32_model --model models/qat_w8/qat_dqn_lander_16_16_seed_0_int8_weights.pt --output firmware/sketch_aug27b/model_data.h

esp_test_32:
	$(PYTHON) -m LunarLander.deployment.export_esp32_model --model models/qat_w8/qat_dqn_lander_32_32_seed_0_int8_weights.pt --output firmware/sketch_aug27b/model_data.h

esp_test_64:
	$(PYTHON) -m LunarLander.deployment.export_esp32_model --model models/qat_w8/qat_dqn_lander_64_64_seed_0_int8_weights.pt --output firmware/sketch_aug27b/model_data.h

esp_test_128:
	$(PYTHON) -m LunarLander.deployment.export_esp32_model --model models/qat_w8/qat_dqn_lander_128_128_seed_0_int8_weights.pt --output firmware/sketch_aug27b/model_data.h

esp_test_256:
	$(PYTHON) -m LunarLander.deployment.export_esp32_model --model models/qat_w8/qat_dqn_lander_256_256_seed_0_int8_weights.pt --output firmware/sketch_aug27b/model_data.h

esp_test_float_16:
	$(PYTHON) -m LunarLander.deployment.export_esp32_model --model models/extended_fp32/my_dqn_lander_16_16_seed_0.zip --output firmware/sketch_aug27b/model_data.h

esp_test_float_32:
	$(PYTHON) -m LunarLander.deployment.export_esp32_model --model models/extended_fp32/my_dqn_lander_32_32_seed_0.zip --output firmware/sketch_aug27b/model_data.h

esp_test_float_64:
	$(PYTHON) -m LunarLander.deployment.export_esp32_model --model models/extended_fp32/my_dqn_lander_64_64_seed_0.zip --output firmware/sketch_aug27b/model_data.h

esp_test_float_128:
	$(PYTHON) -m LunarLander.deployment.export_esp32_model --model models/extended_fp32/my_dqn_lander_128_128_seed_0.zip --output firmware/sketch_aug27b/model_data.h

esp_test_float_256:
	$(PYTHON) -m LunarLander.deployment.export_esp32_model --model models/extended_fp32/my_dqn_lander_256_256_seed_0.zip --output firmware/sketch_aug27b/model_data.h

esp_test:
	$(PYTHON) -m LunarLander.deployment.esp_test --port /dev/cu.usbserial-110 --render

esp_benchmark:
	@test -n "$(MODEL_TYPE)" || (echo "Set MODEL_TYPE=int8 or MODEL_TYPE=float32"; exit 1)
	@test -n "$(SIZE)" || (echo "Set SIZE=16, 32, 64, 128 or 256"; exit 1)
	$(PYTHON) -m LunarLander.deployment.esp_benchmark --port /dev/cu.usbserial-110 --model-type $(MODEL_TYPE) --size $(SIZE) --episodes 10

# make esp_benchmark MODEL_TYPE=int8 SIZE=16
# make esp_benchmark MODEL_TYPE=float32 SIZE=16


# Initial FP32 (models/initial_fp32/) versus W8-only QAT trained from scratch.
# Run sequentially even with make -j: comparison requires completed training.
.PHONY: train_qat_w_scratch compare_scratch scratch_experiment
train_qat_w_scratch:
	$(PYTHON) -m LunarLander.training.train_qat_weights_scratch

compare_scratch:
	$(PYTHON) -m LunarLander.evaluation.compare_scratch

scratch_experiment: train_qat_w_scratch
	$(PYTHON) -m LunarLander.evaluation.compare_scratch

# Run in a separate terminal; stop with Ctrl+C after training.
.PHONY: nosleep
nosleep:
	caffeinate -is

# Regenerate thesis-style scratch tables from saved comparison results.
.PHONY: table_scratch
table_scratch:
	$(PYTHON) -m LunarLander.reporting.generate_scratch_table

.PHONY: lander_train_continues lander_test_continues lander_manual train_lib test_lib base train_base test_base train_qat_w test_qat_w train_qat test_qat test_int_4 compare test_int_w test_int train_qat_4 train_qat_4_w size plot_compare plot_benchmark plot_steps combine_plots inspect benchmark table esp_test_16 esp_test_32 esp_test_64 esp_test_128 esp_test_256 esp_test_float_16 esp_test_float_32 esp_test_float_64 esp_test_float_128 esp_test_float_256 esp_test esp_benchmark train_qat_w_scratch compare_scratch scratch_experiment nosleep table_scratch

.PHONY: help
help:
	@echo "Choose a target from ../README.md; no training is started by default."
