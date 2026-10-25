"""One isolated 64x64 FP32 run using the initial-training protocol."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from LunarLander.paths import APP_ROOT
from LunarLander.training import train_base_initial
from rl_models import dqn


VALIDATE_DIR = APP_ROOT / "models" / "validate"


class _Tee:
    """Keep live console output while recording the same text in the run log."""

    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def main(output_dir=VALIDATE_DIR):
    output_dir = Path(output_dir).resolve()
    # Reserve the entire directory before opening any model, CSV or log files.
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit(
            f"Refusing to overwrite existing validation directory: {output_dir}\n"
            "Move the previous validation run elsewhere before starting a new one."
        ) from None

    manifest_path = output_dir / "run.json"
    manifest = {
        "status": "running",
        "model_type": "FP32",
        "from_scratch": True,
        "architecture": [64, 64],
        "seed": 0,
        "target_steps": 2_500_000,
        "block_size": 25_000,
        "exploration_decay_steps": train_base_initial.EXPLORATION_DECAY_STEPS,
        "eval_seeds": list(train_base_initial.EVAL_SEEDS),
        "protocol": "initial FP32 baseline with fixed 500,000-step exploration decay",
        "output_dir": str(output_dir),
        "started_utc": _utc_now(),
        "source_sha256": {
            Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (train_base_initial, dqn)
        },
    }

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save_manifest()
    with (output_dir / "training.log").open("x", buffering=1) as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            print(f"Validation output: {output_dir}")
            print("One FP32 64x64 model, seed 0, 2,500,000 training steps.")
            print(f"Exploration decay: {train_base_initial.EXPLORATION_DECAY_STEPS:,} steps")
            try:
                train_base_initial.train_models(
                    model_sizes=[64], seeds=[0], output_dir=output_dir,
                )
            except BaseException as error:
                manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                manifest["error"] = f"{type(error).__name__}: {error}"
                manifest["finished_utc"] = _utc_now()
                save_manifest()
                print(f"Validation run {manifest['status']}; outputs retained in {output_dir}")
                raise
            else:
                manifest["status"] = "complete"
                manifest["finished_utc"] = _utc_now()
                save_manifest()
                print(f"Validation training complete. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
