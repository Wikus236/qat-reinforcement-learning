REPO_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PYTHON ?= $(REPO_ROOT)/.venv/bin/python

.PHONY: help
help:
	@echo "LunarLander: make base, train_base, scratch_experiment, compare, benchmark"
	@echo "Reports: make plot_compare, plot_benchmark, table, table_scratch"
	@echo "Pendulum: make pendulum_training_report, pendulum_report_fresh_2m"
	@echo "Use make -C LunarLander <target> or make -C Pendulum <target> for experiment commands."

.PHONY: lander_train_continues lander_test_continues lander_manual train_lib test_lib base train_base test_base train_qat_w test_qat_w train_qat test_qat test_int_4 compare test_int_w test_int train_qat_4 train_qat_4_w size plot_compare plot_benchmark plot_steps combine_plots inspect benchmark table esp_test_16 esp_test_32 esp_test_64 esp_test_128 esp_test_256 esp_test_float_16 esp_test_float_32 esp_test_float_64 esp_test_float_128 esp_test_float_256 esp_test esp_benchmark train_qat_w_scratch compare_scratch scratch_experiment nosleep table_scratch
lander_train_continues lander_test_continues lander_manual train_lib test_lib base train_base test_base train_qat_w test_qat_w train_qat test_qat test_int_4 compare test_int_w test_int train_qat_4 train_qat_4_w size plot_compare plot_benchmark plot_steps combine_plots inspect benchmark table esp_test_16 esp_test_32 esp_test_64 esp_test_128 esp_test_256 esp_test_float_16 esp_test_float_32 esp_test_float_64 esp_test_float_128 esp_test_float_256 esp_test esp_benchmark train_qat_w_scratch compare_scratch scratch_experiment nosleep table_scratch:
	$(MAKE) -C "$(REPO_ROOT)/LunarLander" $@ PYTHON="$(PYTHON)"

.PHONY: pendulum_manual pendulum_export pendulum_export_base pendulum_export_base_best pendulum_export_mild pendulum_export_mild_best pendulum_export_wide pendulum_export_wide_best pendulum_train pendulum_train_both pendulum_train_base pendulum_train_base_overnight pendulum_train_mild pendulum_train_wide pendulum_test pendulum_test_base pendulum_test_mild pendulum_test_wide pendulum_robustness pendulum_robustness_base pendulum_robustness_mild pendulum_robustness_wide pendulum_nosleep pendulum_log_pendulum pendulum_training_report pendulum_train_fresh_2m pendulum_report_fresh_2m
pendulum_manual pendulum_export pendulum_export_base pendulum_export_base_best pendulum_export_mild pendulum_export_mild_best pendulum_export_wide pendulum_export_wide_best pendulum_train pendulum_train_both pendulum_train_base pendulum_train_base_overnight pendulum_train_mild pendulum_train_wide pendulum_test pendulum_test_base pendulum_test_mild pendulum_test_wide pendulum_robustness pendulum_robustness_base pendulum_robustness_mild pendulum_robustness_wide pendulum_nosleep pendulum_log_pendulum pendulum_training_report pendulum_train_fresh_2m pendulum_report_fresh_2m:
	$(MAKE) -C "$(REPO_ROOT)/Pendulum" $(patsubst pendulum_%,%,$@) PYTHON="$(PYTHON)"
