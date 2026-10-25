# Skripsie: Quantization-Aware Reinforcement Learning

Experiments comparing FP32 DQN policies with quantization-aware training (QAT),
using LunarLander and a separate physical Pendulum demonstration.

## Find things

```text
rl_models/                   Shared FP32/QAT learners and exported-model inference
LunarLander/
  training/                  Initial FP32, extended FP32 and QAT training
  evaluation/                Comparison, benchmarking and manual model checks
  plotting/                  Training and benchmark figures
  reporting/                 LaTeX/CSV/Markdown table generators
  deployment/                ESP32 export and serial testing
  firmware/                  Arduino sketch and model header
  models/                    Local policies, training curves and run records
  results/
    data/                    Evaluation and benchmark CSVs
    figures/                 Training, comparison, benchmark and ESP32 plots
    tables/                  Standard report tables
    scratch/                 Separate FP32-versus-from-scratch QAT results
Pendulum/
  environment/               Simulation and manual control
  training/                  Base, refinement, fresh-run training and safeguards
  evaluation/                Policy checks, robustness and simulation logging
  plotting/                  Training and position plots
  deployment/                Model export
  firmware/                  Control, diagnostic, calibration and sensor sketches
  models/original/           Existing base/refinement policies and evaluation logs
  models/fresh_2m/           Separate 2M-step run, logs and its report
  results/                   Original training, hardware and simulation reports
  logs/                      Local console logs (ignored by Git)
docs/                        Report drafts and the old-to-new file map
archive/lunar_lander/         Earlier Stable-Baselines3 prototypes
archive/coursework/          Unrelated coursework, retained locally and ignored
```

## Environment

Commands use the existing project-local Python environment at `.venv/`:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Some workflows also need dependencies not fully covered by the requirements file:

```sh
.venv/bin/python -m pip install "gymnasium[box2d]" pyserial pandas
```

Model binaries and intermediate checkpoints are local files, not distributed in
Git. Evaluating or exporting a policy requires its trained checkpoint. Stored CSV
results and figures can be inspected without retraining.

## LunarLander commands

The existing commands still work from the repository root:

```sh
make base                   # Initial FP32 training: five sizes, five seeds, 2.5M steps each
make train_base             # Additional FP32 training
make train_qat_w             # W8 weights-only QAT refinement
make train_qat               # W8A8 QAT refinement
make train_qat_4_w           # W4 weights-only QAT refinement
make train_qat_4             # W4A4 QAT refinement
make compare
make benchmark
make plot_compare
make plot_benchmark
make table
make scratch_experiment     # Train separate W8 QAT policies from scratch, then compare
make compare_scratch        # Compare completed scratch runs without retraining
make table_scratch
```

These commands are also available through `make -C LunarLander <target>`.
The experiment Makefiles set the Python package path automatically. To run a
Python entry point directly, run it as a module from the repository root, e.g.:

```sh
.venv/bin/python -m LunarLander.evaluation.compare_scratch --curves-only
```

### Single-model validation run

Run `make train_validate` from the repository root to train one fresh **FP32
64x64, seed 0** policy for **2,500,000 steps** using the existing initial baseline
trainer. It preserves the current 25,000-step block/exploration behaviour and
prints the same live training statistics, evaluations and checkpoint progress.

All output goes to `LunarLander/models/validate/`: the best checkpoint,
per-block checkpoints, training-curve CSV, `training.log`, and `run.json` with run
status and configuration. The command refuses to start if that directory already
exists, including after an interrupted run. It does not resume or overwrite it.
This directory is excluded from the standard comparison and benchmark inputs.

To keep the Mac awake, run `make nosleep` in a separate terminal while training.

### LunarLander policies

| Folder under `LunarLander/models/` | Experiment |
| --- | --- |
| `initial_fp32/` | Original FP32 runs trained for 2.5M steps |
| `extended_fp32/` | FP32 models after additional training |
| `qat_w8/`, `qat_w8a8/` | 8-bit QAT refinement |
| `qat_w4/`, `qat_w4a4/` | 4-bit QAT refinement |
| `qat_w8_scratch/` | Separate W8 QAT training from scratch |

Standard comparison and benchmark scripts exclude `qat_w8_scratch/`. Its separate
comparison uses `initial_fp32/`, rather than the extended FP32 policies.

## Pendulum commands

Existing commands continue to work inside `Pendulum/`:

```sh
make -C Pendulum training_report
make -C Pendulum train_fresh_2m
make -C Pendulum report_fresh_2m
make -C Pendulum export_base_best
```

From the root, the same commands have a `pendulum_` prefix:

```sh
make pendulum_training_report
make pendulum_report_fresh_2m
```

The fresh run records evaluations at step zero and every 25,000 steps and refuses
to overwrite an existing run folder. The default exported model remains the
original `dqn_qat_linear_pendulum_base_best.zip`. New training does not automatically
replace the hardware model. Arduino sketches now live under `Pendulum/firmware/`.
Hardware serial ports and uploaded sketch selection depend on the local setup.

## Results and reproducibility

- Computer inference plots and tables display microseconds; source benchmark CSV
  timing remains in milliseconds. ESP32 timing remains in milliseconds.
- NumPy exports rescale stored integer weights for floating-point computation;
  they do not measure native integer-kernel acceleration.
- Simulated INT4 weights use INT8 storage. Additional packed-INT4 memory savings
  are theoretical estimates.
- Original Pendulum base logs begin partway through training; the separate fresh
  2M-step run records history from step zero.
- Historical CSV cells and console output may mention paths from before the
  reorganization. Raw experimental records were preserved unchanged.

See [the reorganization guide](docs/REORGANIZATION.md) for old and new paths.
Intermediate checkpoints, binary policies, logs, credentials and coursework are
ignored for future Git additions. Evaluation CSVs, plots, tables and Arduino model
headers remain available. Ignore rules do not erase older Git history.

Some model loaders use unrestricted PyTorch checkpoint deserialization. Only load
model files from a trusted source.
