"""Back up and replace selected LunarLander sizes, training types and seeds."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

from LunarLander.paths import APP_ROOT

FAMILIES = (
    ("initial_fp32", "train_base_initial"),
    ("extended_fp32", "train_base_extended"),
    ("qat_w8", "train_qat_weights8"),
    ("qat_w8a8", "train_qat8"),
    ("qat_w4", "train_qat_weights4"),
    ("qat_w4a4", "train_qat4"),
    ("qat_w8_scratch", "train_qat_weights_scratch"),
)


def resolve_families(names=None):
    if names is None:
        return FAMILIES
    unknown = set(names) - {name for name, _ in FAMILIES}
    if unknown:
        raise ValueError(f"Unknown training types: {sorted(unknown)}")
    return tuple(pair for pair in FAMILIES if pair[0] in names)


def missing_prerequisites(root, sizes, families, seeds):
    if "initial_fp32" in families or not any(
        name not in ("initial_fp32", "qat_w8_scratch") for name in families
    ):
        return []
    return [path for size in sizes for seed in seeds if not (
        path := Path(root) / "models/initial_fp32" / f"my_dqn_lander_{size}_{size}_seed_{seed}.zip"
    ).is_file()]


def selected_artifacts(root, sizes, families=None, seeds=None):
    """Find selected policies, exports, curves, manifests and checkpoint folders."""
    paths = []
    for family, _ in resolve_families(families):
        directory = Path(root) / "models" / family
        for size in sizes:
            for seed in (range(5) if seeds is None else seeds):
                suffix = f"{size}_{size}_seed_{seed}"
                # Root files have a known policy prefix; nested artifacts have
                # either this exact directory name or a curve prefix.
                if directory.exists():
                    for path in directory.iterdir():
                        if path.is_file() and any(
                            path.name.endswith(f"_{suffix}{ending}")
                            for ending in (".zip", "_int8_weights.pt", "_int4_weights.pt")
                        ):
                            paths.append(path)
                for parent in ("checkpoints", "qat_checkpoints", "training_curves", "runs"):
                    folder = directory / parent
                    if not folder.exists():
                        continue
                    paths.extend(path for path in folder.iterdir() if (
                        path.name == suffix
                        or path.name == f"{suffix}.json"
                        or path.name.endswith(f"_{suffix}.csv")
                    ))
    return sorted(set(paths))


def preserve_previous(root, sizes, backup, families=None, seeds=None):
    root, backup = Path(root), Path(backup)
    backup.mkdir(parents=True, exist_ok=False)
    # Snapshot reports before replacing policies; hardware data is copied only.
    if (root / "results").exists():
        shutil.copytree(root / "results", backup / "results")
    paths = selected_artifacts(root, sizes, families, seeds)
    for path in paths:
        destination = backup / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), destination)
    return len(paths)


def training_commands(sizes, python=sys.executable, families=None, seeds=None):
    return [
        [python, "-u", "-m", f"LunarLander.training.{module}",
         "--sizes", *map(str, sizes),
         *([] if seeds is None else ["--seeds", *map(str, seeds)])]
        for _, module in resolve_families(families)
    ]


def run_logged(command, cwd, log):
    print("Running: " + " ".join(command), flush=True)
    with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", required=True,
                        choices=(16, 32, 64, 128, 256))
    parser.add_argument("--types", nargs="+", choices=[name for name, _ in FAMILIES],
                        default=[name for name, _ in FAMILIES],
                        help="Training types to replace; default: all seven types")
    parser.add_argument("--seeds", nargs="+", type=int, choices=range(5),
                        default=list(range(5)), help="Training seeds; default: 0 1 2 3 4")
    args = parser.parse_args()
    sizes = sorted(set(args.sizes))
    seeds = sorted(set(args.seeds))
    families = [name for name, _ in resolve_families(args.types)]
    root = Path(APP_ROOT)
    repo = root.parent
    missing = missing_prerequisites(root, sizes, families, seeds)
    if missing:
        parser.error("Missing initial FP32 policies; include initial_fp32 in --types or train them first:\n"
                     + "\n".join(map(str, missing)))
    tag = "_".join(map(str, sizes))
    backup = root / "reruns" / f"sizes_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    count = preserve_previous(root, sizes, backup, families, seeds)
    state = dict(sizes=sizes, seeds=seeds, types=families, status="training",
                 preserved_artifacts=count, exploration_decay_steps=500_000,
                 additional_initial_epsilon=0.4, additional_final_epsilon=0.01)
    manifest = backup / "run.json"
    manifest.write_text(json.dumps(state, indent=2) + "\n")
    print(f"Previous models and reports preserved in: {backup}", flush=True)
    print(f"Training {len(sizes) * len(seeds) * len(families)} runs; original budgets/settings retained.", flush=True)
    print(f"Sizes: {sizes}; types: {families}; seeds: {seeds}", flush=True)
    try:
        with (backup / "training.log").open("w", buffering=1) as log:
            for command in training_commands(sizes, families=families, seeds=seeds):
                run_logged(command, repo, log)
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        manifest.write_text(json.dumps(state, indent=2) + "\n")
    print(f"Selected retraining complete. Previous run: {backup}", flush=True)
    print("Run make lunarlander_reports to regenerate all computer comparisons, plots and tables.", flush=True)


if __name__ == "__main__":
    main()
