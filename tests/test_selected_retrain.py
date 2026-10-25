"""Check selected-width isolation and backup before running expensive training."""
import tempfile
from pathlib import Path
import unittest

from LunarLander.training.retrain_selected import preserve_previous, training_commands, FAMILIES
from LunarLander.training.settings import selected_training_sizes


class SelectedRetrainTests(unittest.TestCase):
    def test_size_arguments_and_defaults(self):
        self.assertEqual(selected_training_sizes([16, 32, 64], []), [16, 32, 64])
        self.assertEqual(selected_training_sizes([16, 32, 64], ["--sizes", "64"]), [64])
        with self.assertRaises(SystemExit):
            selected_training_sizes([16, 32, 64], ["--sizes", "256"])

    def test_backup_isolates_selected_width_and_preserves_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "LunarLander"
            selected, untouched = [], []
            for family, _ in FAMILIES:
                for size, collection in ((64, selected), (128, untouched)):
                    for name in (f"policy_{size}_{size}_seed_0.zip",
                                 f"policy_{size}_{size}_seed_0_int8_weights.pt",
                                 f"training_curves/curve_{size}_{size}_seed_0.csv",
                                 f"runs/{size}_{size}_seed_0.json",
                                 f"checkpoints/{size}_{size}_seed_0/old.zip",
                                 f"qat_checkpoints/{size}_{size}_seed_0/old.zip"):
                        path=root / "models" / family / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(family + name)
                        collection.append((path, path.read_text()))
            report=root / "results/data/esp_benchmark_results.csv"
            report.parent.mkdir(parents=True)
            report.write_text("original hardware measurements")
            backup=root / "reruns/test"
            preserve_previous(root, [64], backup)
            for path, value in selected:
                self.assertFalse(path.exists())
                self.assertEqual((backup/path.relative_to(root)).read_text(), value)
            for path, value in untouched:
                self.assertEqual(path.read_text(), value)
            self.assertEqual(report.read_text(), "original hardware measurements")
            self.assertEqual((backup/"results/data/esp_benchmark_results.csv").read_text(), report.read_text())

    def test_all_seven_families_receive_only_selected_size(self):
        commands=training_commands([64], python="python")
        self.assertEqual(len(commands), 7)
        self.assertEqual([c[3].split(".")[-1] for c in commands], [m for _,m in FAMILIES])
        self.assertTrue(all(c[-2:]==["--sizes", "64"] for c in commands))




class SpecificModelTests(unittest.TestCase):
    def test_config_restricts_both_sizes_and_seeds(self):
        from LunarLander.training.settings import selected_training_config
        selection=selected_training_config([16,32,64], [0,1,2,3,4],
                                           ['--sizes','64','32','64','--seeds','2'])
        self.assertEqual(selection.sizes, [32,64])
        self.assertEqual(selection.seeds, [2])

    def test_only_requested_type_and_seed_are_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            selected=[]
            untouched=[]
            for family, _ in FAMILIES:
                for size in [32,64]:
                    for seed in [0,2]:
                        path=root/'models'/family/f'policy_{size}_{size}_seed_{seed}.zip'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(f'{family}/{size}/{seed}')
                        (selected if (family,size,seed)==('qat_w8',64,2) else untouched).append(path)
            backup=root/'reruns/test'
            count=preserve_previous(root,[64],backup,families=['qat_w8'],seeds=[2])
            self.assertEqual(count,1)
            self.assertTrue(all(p.exists() for p in untouched))
            self.assertFalse(selected[0].exists())
            self.assertTrue((backup/selected[0].relative_to(root)).exists())

    def test_type_order_and_seed_filter(self):
        commands=training_commands([64],python='python',
                                  families=['qat_w4','initial_fp32'], seeds=[2])
        self.assertEqual([c[3].split('.')[-1] for c in commands],
                         ['train_base_initial','train_qat_weights4'])
        self.assertTrue(all(c[-4:]==['--sizes','64','--seeds','2'] for c in commands))

    def test_prerequisites_before_backup(self):
        from LunarLander.training.retrain_selected import missing_prerequisites
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(len(missing_prerequisites(root,[64],['qat_w8'],[2])),1)
            self.assertEqual(missing_prerequisites(root,[64],['initial_fp32','qat_w8'],[2]),[])
            self.assertEqual(missing_prerequisites(root,[64],['qat_w8_scratch'],[2]),[])
            policy=root/'models/initial_fp32/my_dqn_lander_64_64_seed_2.zip'
            policy.parent.mkdir(parents=True)
            policy.touch()
            self.assertEqual(missing_prerequisites(root,[64],['qat_w8'],[2]),[])

    def test_make_passes_explicit_selection(self):
        import subprocess
        repo=Path(__file__).resolve().parents[1]
        result=subprocess.run(['make','-n','retrain_lunarlander_selected',
                               'SIZES=32 64','TYPES=qat_w8','SEEDS=2'],
                              cwd=repo,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--sizes 32 64 --types qat_w8 --seeds 2',result.stdout)


if __name__ == "__main__":
    unittest.main()
