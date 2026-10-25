"""Check the unattended pipeline order and completeness guard without training."""

import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from LunarLander.evaluation.validate_experiment import SPECS, validate_experiment

ROOT = Path(__file__).resolve().parents[1]
TRAINING = [
    "train_base_initial", "train_base_extended", "train_qat_weights8", "train_qat8",
    "train_qat_weights4", "train_qat4", "train_qat_weights_scratch",
]
REPORTS = [
    "LunarLander.evaluation.validate_experiment", "LunarLander.evaluation.compare",
    "LunarLander.evaluation.benchmark", "LunarLander.evaluation.compare_scratch",
    "LunarLander.plotting.plot_steps", "LunarLander.plotting.combine_plots",
    "LunarLander.plotting.plot_compare", "LunarLander.plotting.plot_benchmark",
    "LunarLander.reporting.generate_table_data", "LunarLander.reporting.generate_scratch_table",
]


class PipelineTests(unittest.TestCase):
    def run_fake_pipeline(self, fail=""):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            log = folder / "calls.jsonl"
            python = folder / "fake_python"
            python.write_text(
                "#!/usr/bin/env python3\nimport json,os,sys\n"
                "module=sys.argv[sys.argv.index('-m')+1]\n"
                "with open(os.environ['PIPELINE_TEST_LOG'],'a') as f:\n"
                " f.write(json.dumps({'module':module,'args':sys.argv,'backend':os.getenv('MPLBACKEND')})+'\\n')\n"
                "sys.exit(17 if module==os.getenv('PIPELINE_TEST_FAIL') else 0)\n"
            )
            python.chmod(0o755)
            env = {**os.environ, "PIPELINE_TEST_LOG": str(log), "PIPELINE_TEST_FAIL": fail}
            result = subprocess.run(
                ["make", "-j8", "-C", str(ROOT / "LunarLander"), "retrain_lunarlander",
                 f"PYTHON={python}"], env=env, text=True, capture_output=True,
            )
            return result, [json.loads(row) for row in log.read_text().splitlines()]

    def test_all_stages_run_in_order_even_with_parallel_make(self):
        result, calls = self.run_fake_pipeline()
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = ["LunarLander.training." + name for name in TRAINING] + REPORTS
        self.assertEqual([row["module"] for row in calls], expected)
        self.assertIn("--overwrite", calls[6]["args"])
        self.assertTrue(all(row["backend"] == "Agg" for row in calls[7:]))
        self.assertFalse(any("esp" in row["module"] for row in calls))

    def test_failed_stage_stops_before_later_reports(self):
        result, calls = self.run_fake_pipeline("LunarLander.evaluation.compare")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls[-1]["module"], "LunarLander.evaluation.compare")
        self.assertNotIn("LunarLander.evaluation.benchmark", [row["module"] for row in calls])

    def test_complete_experiment_and_wrong_schedule_detection(self):
        from LunarLander.evaluation import validate_experiment as check
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # The same validator loops all widths/seeds in production; a small matrix
            # tests its artifact and schedule requirements without model training.
            with patch.object(check, "MODEL_SIZES", [16]), patch.object(check, "SEEDS", [0]):
                for name, prefix, curve_prefix, budget, export in SPECS:
                    directory = root / "models" / name
                    directory.mkdir(parents=True)
                    stem = f"{prefix}_16_16_seed_0"
                    torch.save({"seed": 0, "exploration_decay_steps": 500_000,
                                "exploration_initial_eps": 1.0 if budget == 2_500_000 else 0.4,
                                "exploration_final_eps": 0.05 if budget == 2_500_000 else 0.01,
                                "policy_kwargs": {"net_arch": [16, 16]}}, directory / f"{stem}.zip")
                    curves = directory / "training_curves"
                    curves.mkdir()
                    with (curves / f"{curve_prefix}_16_16_seed_0.csv").open("w", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(["size", "train_seed", "training_steps", "mean_reward"])
                        writer.writerows([16, 0, step, 1] for step in range(0, budget + 1, 25_000))
                    if export:
                        (directory / f"{stem}_{export}_weights.pt").write_bytes(b"fixture")
                    if name == "qat_w8_scratch":
                        (directory / "runs").mkdir()
                        (directory / "runs/16_16_seed_0.json").write_text(json.dumps(
                            {"completed": True, "exploration_decay_steps": 500_000, "total_steps": budget}))
                validate_experiment(root)
                path = root / "models/extended_fp32/my_dqn_lander_16_16_seed_0.zip"
                checkpoint = torch.load(path, weights_only=True)
                checkpoint["exploration_initial_eps"] = 1.0
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ValueError, "Wrong exploration schedule"):
                    validate_experiment(root)
                checkpoint["exploration_initial_eps"] = 0.4
                torch.save(checkpoint, path)
                (root / "models/qat_w8/qat_dqn_lander_16_16_seed_0_int8_weights.pt").unlink()
                with self.assertRaisesRegex(ValueError, "Missing export"):
                    validate_experiment(root)


if __name__ == "__main__":
    unittest.main()
