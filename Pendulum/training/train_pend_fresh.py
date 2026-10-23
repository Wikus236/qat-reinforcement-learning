"""Train a separate 2M-step Pendulum base QAT model and log from step zero.

Reuses the existing base trainer, environment, hyperparameters and safeguards.
Refuses existing output folders so a fresh run cannot resume or overwrite models.
"""

from Pendulum.paths import APP_ROOT
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import signal
from types import SimpleNamespace

import numpy as np
import torch

from Pendulum.training import train_pend_qat as training
from Pendulum.plotting.plot_training_pendulum import read_training, plot_training, summary_rows, write_table

ROOT = APP_ROOT
DEFAULT_OUTPUT = APP_ROOT / "models/fresh_2m"


def generate_report(output_dir):
    logs = {"base": read_training(output_dir / "dqn_qat_linear_pendulum_base_training.csv", "base")}
    report_dir = output_dir / "training_report"
    report_dir.mkdir(exist_ok=True)
    plot_training(logs, report_dir)
    write_table(summary_rows(logs), report_dir, logs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--report-only", action="store_true",
                        help="Refresh plots and table from an existing fresh-run CSV")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if args.report_only:
        generate_report(output)
        return
    if min(args.steps, args.eval_freq, args.eval_episodes) <= 0:
        parser.error("steps, eval-freq and eval-episodes must be positive")
    if output == DEFAULT_OUTPUT and args.steps != 2_000_000:
        parser.error("Use a separate --output-dir for nonstandard runs")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output already exists: {output}. Choose a new folder; existing runs are never loaded.")

    # Redirect only this process's base configuration, never the original files.
    training.MODEL_DIR = output
    config = dict(training.PHASES["base"])
    config.update(model_path=output / "dqn_qat_linear_pendulum_base", start_paths=[])
    training.PHASES = {**training.PHASES, "base": config}
    training.LOAD_EXISTING_MODEL = False
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    run = dict(status="running", from_scratch=True, target_steps=args.steps, seed=42,
               architecture=[128, 128], quantization="8-bit weights only",
               eval_freq=args.eval_freq, eval_episodes=args.eval_episodes,
               started_utc=datetime.now(timezone.utc).isoformat())
    manifest = output / "run.json"
    manifest.write_text(json.dumps(run, indent=2) + "\n")
    print(f"Fresh run: {output}; target {args.steps:,} steps; seed 42", flush=True)

    def stop(signum, frame):
        raise KeyboardInterrupt("Training interrupted")
    previous = signal.signal(signal.SIGTERM, stop)
    exit_code = 0
    try:
        training.train_phase(SimpleNamespace(seed=42, resume=False,
                             eval_freq=args.eval_freq, eval_episodes=args.eval_episodes),
                             "base", args.steps)
        run["status"] = "complete"
    except KeyboardInterrupt:
        run["status"] = "interrupted"
        exit_code = 130
    except Exception as error:
        run.update(status="failed", error=str(error))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        run["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest.write_text(json.dumps(run, indent=2) + "\n")
        if (output / "dqn_qat_linear_pendulum_base_training.csv").exists():
            generate_report(output)
        print(f"Run status: {run['status']}; outputs: {output}", flush=True)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
