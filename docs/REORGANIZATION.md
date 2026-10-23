# Reorganization guide

Makefile command names are preserved. Run Python entry points as modules from the repository root.

## Folder moves

| Previous path | New path |
| --- | --- |
| `policies/` | `LunarLander/models/initial_fp32/` |
| `policies_base/` | `LunarLander/models/extended_fp32/` |
| `policies_qat/` | `LunarLander/models/qat_w8a8/` |
| `policies_qat_4/` | `LunarLander/models/qat_w4a4/` |
| `policies_qat_w/` | `LunarLander/models/qat_w8/` |
| `policies_qat_w_4/` | `LunarLander/models/qat_w4/` |
| `policies_qat_w_scratch/` | `LunarLander/models/qat_w8_scratch/` |
| `plots/` | `LunarLander/results/figures/training/` |
| `benchmark_plots/` | `LunarLander/results/figures/benchmark/` |
| `comparison_plots/` | `LunarLander/results/figures/comparison/` |
| `esp_benchmark_plots/` | `LunarLander/results/figures/esp32/` |
| `table_data/` | `LunarLander/results/tables/` |
| `comparison_scratch/` | `LunarLander/results/scratch/` |
| `Arduino/` | `LunarLander/firmware/` |
| `Compare/` | `docs/drafts/` |
| `DQN/` | `archive/coursework/` |
| `Pendulum/Arduino/` | `Pendulum/firmware/` |
| `Pendulum/policies_pend_qat/` | `Pendulum/models/original/` |
| `Pendulum/policies_pend_qat_fresh_2m/` | `Pendulum/models/fresh_2m/` |
| `Pendulum/hardware_logs/` | `Pendulum/results/hardware/` |
| `Pendulum/pendulum_logs/` | `Pendulum/results/simulation/` |
| `Pendulum/training_report/` | `Pendulum/results/training/` |

## Source and data file moves

| Previous path | New path |
| --- | --- |
| `DQN_base.py` | `rl_models/dqn.py` |
| `DQN_QAT.py` | `rl_models/qat.py` |
| `DQN_QAT_4.py` | `rl_models/qat4.py` |
| `DQN_QAT_Weights.py` | `rl_models/qat_weights.py` |
| `DQN_QAT_Weights_4.py` | `rl_models/qat_weights4.py` |
| `deployment_inference.py` | `rl_models/inference.py` |
| `base.py` | `LunarLander/training/train_base_initial.py` |
| `Train_base.py` | `LunarLander/training/train_base_extended.py` |
| `Train_QAT.py` | `LunarLander/training/train_qat8.py` |
| `Train_QAT_4.py` | `LunarLander/training/train_qat4.py` |
| `Train_QAT_W.py` | `LunarLander/training/train_qat_weights8.py` |
| `Train_QAT_W_4.py` | `LunarLander/training/train_qat_weights4.py` |
| `Train_QAT_W_scratch.py` | `LunarLander/training/train_qat_weights_scratch.py` |
| `compare.py` | `LunarLander/evaluation/compare.py` |
| `compare_scratch.py` | `LunarLander/evaluation/compare_scratch.py` |
| `benchmark.py` | `LunarLander/evaluation/benchmark.py` |
| `size_compare.py` | `LunarLander/evaluation/size_compare.py` |
| `esp_benchmark.py` | `LunarLander/deployment/esp_benchmark.py` |
| `esp_benchmark_data.py` | `LunarLander/deployment/esp_benchmark_data.py` |
| `esp_test.py` | `LunarLander/deployment/esp_test.py` |
| `export_esp32_model.py` | `LunarLander/deployment/export_esp32_model.py` |
| `inspect_weights.py` | `LunarLander/evaluation/inspect_weights.py` |
| `Lunar_Manual.py` | `LunarLander/evaluation/manual.py` |
| `plot_compare.py` | `LunarLander/plotting/plot_compare.py` |
| `plot_benchmark.py` | `LunarLander/plotting/plot_benchmark.py` |
| `plot_steps.py` | `LunarLander/plotting/plot_steps.py` |
| `combine_plots.py` | `LunarLander/plotting/combine_plots.py` |
| `plot_esp_benchmark.py` | `LunarLander/plotting/plot_esp_benchmark.py` |
| `generate_table_data.py` | `LunarLander/reporting/generate_table_data.py` |
| `generate_scratch_table.py` | `LunarLander/reporting/generate_scratch_table.py` |
| `generate_esp_table.py` | `LunarLander/reporting/generate_esp_table.py` |
| `Pendulum/train_pend_qat.py` | `Pendulum/training/train_pend_qat.py` |
| `Pendulum/train_pend_fresh.py` | `Pendulum/training/train_pend_fresh.py` |
| `Pendulum/base_training_guard.py` | `Pendulum/training/base_training_guard.py` |
| `Pendulum/mild_dqn.py` | `Pendulum/training/mild_dqn.py` |
| `Pendulum/pendulum_env.py` | `Pendulum/environment/pendulum_env.py` |
| `Pendulum/manual_pendulum.py` | `Pendulum/environment/manual_pendulum.py` |
| `Pendulum/test_pend_qat.py` | `Pendulum/evaluation/test_pend_qat.py` |
| `Pendulum/evaluate_pend_robustness.py` | `Pendulum/evaluation/evaluate_pend_robustness.py` |
| `Pendulum/log_pendulum.py` | `Pendulum/evaluation/log_pendulum.py` |
| `Pendulum/export_pendulum_model.py` | `Pendulum/deployment/export_pendulum_model.py` |
| `Pendulum/plot_hardware_pendulum.py` | `Pendulum/plotting/plot_hardware_pendulum.py` |
| `Pendulum/position_plot.py` | `Pendulum/plotting/position_plot.py` |
| `Pendulum/plot_training_pendulum.py` | `Pendulum/plotting/plot_training_pendulum.py` |
| `Test_base.py` | `LunarLander/evaluation/manual_tests/test_base.py` |
| `Test_QAT.py` | `LunarLander/evaluation/manual_tests/test_qat.py` |
| `Test_QAT_W.py` | `LunarLander/evaluation/manual_tests/test_qat_w.py` |
| `Test_int.py` | `LunarLander/evaluation/manual_tests/test_int.py` |
| `Test_int_4.py` | `LunarLander/evaluation/manual_tests/test_int_4.py` |
| `Test_int_W.py` | `LunarLander/evaluation/manual_tests/test_int_w.py` |
| `other_models/Lunar_Test_Continues.py` | `archive/lunar_lander/Lunar_Test_Continues.py` |
| `other_models/Lunar_Train_Continues.py` | `archive/lunar_lander/Lunar_Train_Continues.py` |
| `other_models/Lunar_Train.py` | `archive/lunar_lander/Lunar_Train.py` |
| `other_models/Lunar_Test.py` | `archive/lunar_lander/Lunar_Test.py` |
| `model_benchmark_results.csv` | `LunarLander/results/data/model_benchmark_results.csv` |
| `model_results_summary.csv` | `LunarLander/results/data/model_results_summary.csv` |
| `esp_benchmark_results.csv` | `LunarLander/results/data/esp_benchmark_results.csv` |
| `model_results_detailed.csv` | `LunarLander/results/data/model_results_detailed.csv` |

No trained models, CSV values, plots, firmware contents or Git history were discarded. Local coursework and intermediate checkpoints remain ignored. The original raw data may contain old file paths.

The default Pendulum export remains the original best base checkpoint; the fresh two-million-step experiment remains separate.
