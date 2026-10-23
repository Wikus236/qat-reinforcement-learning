"""Create report figures and a LaTeX summary from saved Pendulum evaluations.

Run from the repository root with make pendulum_training_report, or use make inside Pendulum.
No training, checkpoint loading or hardware connection is performed.
"""

from Pendulum.paths import APP_ROOT
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = APP_ROOT
METRICS = ("mean_reward", "reward_std", "capture_rate", "balance_fraction",
           "longest_balance_seconds", "rail_failure_rate")


def read_training(path, phase):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty training log: {path}")
    parsed = []
    for row in rows:
        if row["phase"] != phase:
            raise ValueError(f"Unexpected phase in {path}")
        parsed.append({**row, "steps": int(row["steps"]),
                       **{key: float(row[key]) for key in METRICS}})
    steps = np.array([row["steps"] for row in parsed])
    if np.any(np.diff(steps) <= 0):
        raise ValueError(f"Steps must increase strictly in {path}; separate restarted runs first")
    if not all(np.isfinite(row[key]) for row in parsed for key in METRICS):
        raise ValueError(f"Non-finite metrics in {path}")
    return parsed


def plot_training(logs, output_dir):
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 20,
                         "axes.labelsize": 19, "xtick.labelsize": 17,
                         "ytick.labelsize": 17, "legend.fontsize": 15,
                         "figure.titlesize": 22})
    fig, axes = plt.subplots(2, len(logs), figsize=(8 * len(logs), 10),
                             squeeze=False, sharex="col")
    for column, (phase, rows) in enumerate(logs.items()):
        steps = np.array([row["steps"] for row in rows]) / 1e6
        reward = np.array([row["mean_reward"] for row in rows])
        std = np.array([row["reward_std"] for row in rows])
        top, bottom = axes[:, column]
        top.plot(steps, reward, color="tab:blue", linewidth=1.2, label="Evaluation mean")
        top.fill_between(steps, reward - std, reward + std, color="tab:blue",
                         alpha=0.16, label="Episode reward +/- SD")
        peak = int(np.argmax(reward))
        top.scatter(steps[peak], reward[peak], color="black", marker="*", s=130,
                    zorder=5, label="Highest logged mean")
        subtitle = (f"Base training (log starts at {steps[0]:.2f}M steps)"
                    if phase == "base" else "Mild refinement (phase-local steps)")
        top.set_title(subtitle)
        top.set_ylabel("Evaluation reward")
        top.legend(loc="lower right")
        bottom.plot(steps, [100 * row["capture_rate"] for row in rows],
                    label="Capture rate", color="tab:green", linewidth=1.2)
        bottom.plot(steps, [100 * row["balance_fraction"] for row in rows],
                    label="Balanced time", color="tab:orange", linewidth=1.2)
        bottom.set(xlabel="Training steps (millions)", ylabel="Evaluation performance (%)",
                   ylim=(-3, 103))
        bottom.legend(loc="lower right")
        for ax in (top, bottom):
            ax.grid(alpha=0.25)
            ax.set_xlim(steps[0], steps[-1])
    fig.suptitle("Pendulum QAT training: recorded evaluation history")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"pendulum_training.{extension}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def summary_rows(logs):
    result = []
    for phase, rows in logs.items():
        highest = max(rows, key=lambda row: row["mean_reward"])
        for name, row in (("First recorded", rows[0]), ("Highest mean reward", highest),
                          ("Final recorded", rows[-1])):
            result.append({"phase": phase, "record": name, "steps": row["steps"],
                           **{key: row[key] for key in METRICS}})
    return result


def write_table(rows, output_dir, logs):
    with (output_dir / "pendulum_training_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    log_note = ""
    if "base" in logs:
        log_note = (f" Base logging starts at {logs['base'][0]['steps']:,} steps; "
                    "earlier evaluations are unavailable. Mild uses a separate step counter.")
    lines = [
        r"% Requires: \usepackage{pdflscape,booktabs,adjustbox}",
        r"\begin{landscape}", r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Recorded Pendulum QAT training evaluations. Reward SD is across evaluation episodes. Capture requires at least one second of continuous balance; balanced time is the mean fraction of episode steps spent balanced. The highest logged mean reward is not necessarily the checkpoint selected for deployment." + log_note + "}",
        r"\label{tab:pendulum-training-summary}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{llrrrrrrr}", r"\toprule",
        r"Phase & Record & Steps & Mean reward $\uparrow$ & Reward SD & Capture (\%) $\uparrow$ & Balanced (\%) $\uparrow$ & Mean longest balance (s) $\uparrow$ & Rail failures (\%) $\downarrow$ \\",
        r"\midrule",
    ]
    previous = None
    for row in rows:
        phase = row["phase"]
        if previous is not None and phase != previous:
            lines.append(r"\midrule")
        reward = f"{row['mean_reward']:.2f}"
        if row["record"] == "Highest mean reward":
            reward = r"\textbf{" + reward + "}"
        cells = [phase.capitalize(), row["record"], f"{row['steps']:,}", reward,
                 f"{row['reward_std']:.2f}", f"{100 * row['capture_rate']:.2f}",
                 f"{100 * row['balance_fraction']:.2f}",
                 f"{row['longest_balance_seconds']:.2f}",
                 f"{100 * row['rail_failure_rate']:.2f}"]
        lines.append(" & ".join(cells) + r" \\")
        previous = phase
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
              r"\end{table}", r"\end{landscape}"]
    (output_dir / "pendulum_training_summary.tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=APP_ROOT / "models/original")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "results/training")
    parser.add_argument("--phases", nargs="+", choices=("base", "mild"), default=["base", "mild"])
    args = parser.parse_args()
    try:
        logs = {phase: read_training(args.log_dir / f"dqn_qat_linear_pendulum_{phase}_training.csv", phase)
                for phase in args.phases}
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + "\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_training(logs, args.output_dir)
    write_table(summary_rows(logs), args.output_dir, logs)
    (args.output_dir / "README.txt").write_text(
        "Generated from saved training-evaluation CSVs, without retraining.\n"
        "Figure: evaluation mean reward with +/- one episode-reward SD, capture rate, and balanced time.\n"
        "No smoothing or interpolation; lines connect logged evaluations.\n"
        "These are evaluations of one training run per phase, not averages across training seeds.\n"
        "Base history includes recovery/restoration and exploration-boost interventions recorded in the source CSV.\n"
        "Base logging starts partway through training. Mild has a separate phase-local step counter.\n"
        "Highest mean reward is descriptive and is not the deployment selection rule, which prioritizes sustained balance.\n"
        "The table lists the first, highest-mean-reward and final logged evaluations for each phase.\n"
        "Capture means at least one second of continuous balance. Mean longest balance averages each episode's longest streak.\n"
        "CSV rates remain fractions; plots and LaTeX convert them to percentages.\n"
        "Include the PDF figure with includegraphics and the .tex table with input in your report.\n")
    print(f"Saved report figure (PNG/PDF), summary table (LaTeX/CSV), and notes to {args.output_dir}")


if __name__ == "__main__":
    main()
