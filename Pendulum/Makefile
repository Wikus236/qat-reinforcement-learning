.DEFAULT_GOAL := help
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
PYTHON ?= $(REPO_ROOT)/.venv/bin/python
export PYTHONPATH := $(REPO_ROOT)$(if $(PYTHONPATH),:$(PYTHONPATH))

manual:
	$(PYTHON) -m Pendulum.environment.manual_pendulum

export:
	$(PYTHON) -m Pendulum.deployment.export_pendulum_model

export_base:
	$(PYTHON) -m Pendulum.deployment.export_pendulum_model --model models/original/dqn_qat_linear_pendulum_base.zip

export_base_best:
	$(PYTHON) -m Pendulum.deployment.export_pendulum_model --model models/original/dqn_qat_linear_pendulum_base_best.zip

export_mild:
	$(PYTHON) -m Pendulum.deployment.export_pendulum_model --model models/original/dqn_qat_linear_pendulum_mild_random.zip

export_mild_best:
	$(PYTHON) -m Pendulum.deployment.export_pendulum_model --model models/original/dqn_qat_linear_pendulum_mild_random_best.zip

export_wide:
	$(PYTHON) -m Pendulum.deployment.export_pendulum_model --model models/original/dqn_qat_linear_pendulum_wide_random.zip

export_wide_best:
	$(PYTHON) -m Pendulum.deployment.export_pendulum_model --model models/original/dqn_qat_linear_pendulum_wide_random_best.zip

train:
	$(PYTHON) -m Pendulum.training.train_pend_qat --phase both

train_both:
	$(PYTHON) -m Pendulum.training.train_pend_qat --phase both

train_base:
	$(PYTHON) -m Pendulum.training.train_pend_qat --phase base

# Keeps the Mac awake while the command runs; preserves console output.
train_base_overnight:
	mkdir -p logs
	caffeinate -i $(PYTHON) -u -m Pendulum.training.train_pend_qat --phase base >> logs/base_training.log 2>&1

train_mild:
	$(PYTHON) -m Pendulum.training.train_pend_qat --phase mild

train_wide:
	$(PYTHON) -m Pendulum.training.train_pend_qat --phase wide

test:
	$(PYTHON) -m Pendulum.evaluation.test_pend_qat --phase mild

test_base:
	$(PYTHON) -m Pendulum.evaluation.test_pend_qat --phase base

test_mild:
	$(PYTHON) -m Pendulum.evaluation.test_pend_qat --phase mild

test_wide:
	$(PYTHON) -m Pendulum.evaluation.test_pend_qat --phase wide

robustness:
	$(PYTHON) -m Pendulum.evaluation.evaluate_pend_robustness --phase mild

robustness_base:
	$(PYTHON) -m Pendulum.evaluation.evaluate_pend_robustness --phase base

robustness_mild:
	$(PYTHON) -m Pendulum.evaluation.evaluate_pend_robustness --phase mild

robustness_wide:
	$(PYTHON) -m Pendulum.evaluation.evaluate_pend_robustness --phase wide

nosleep:
	caffeinate -di

.PHONY: log_pendulum

log_pendulum:
	$(PYTHON) -m Pendulum.evaluation.log_pendulum

# Report figure and LaTeX table from saved training evaluations.
.PHONY: training_report
training_report:
	$(PYTHON) -m Pendulum.plotting.plot_training_pendulum

# New fixed-physics QAT model; starts at zero and refuses existing output folders.
.PHONY: train_fresh_2m report_fresh_2m
train_fresh_2m:
	$(PYTHON) -u -m Pendulum.training.train_pend_fresh

report_fresh_2m:
	$(PYTHON) -m Pendulum.training.train_pend_fresh --report-only

.PHONY: manual export export_base export_base_best export_mild export_mild_best export_wide export_wide_best train train_both train_base train_base_overnight train_mild train_wide test test_base test_mild test_wide robustness robustness_base robustness_mild robustness_wide nosleep log_pendulum training_report train_fresh_2m report_fresh_2m

.PHONY: help
help:
	@echo "Choose a target from ../README.md; no training is started by default."
