"""Regression checks for fixed LunarLander decay and legacy learner compatibility."""

import ast
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch

from LunarLander.training.settings import EXPLORATION_DECAY_STEPS
from rl_models.dqn import DQN
from rl_models.qat import DQN_QAT
from rl_models.qat4 import DQN_QAT_4 as DQN_QAT4
from rl_models.qat_weights import DQN_QAT_W
from rl_models.qat_weights4 import DQN_QAT_W_4 as DQN_QAT_W4


LEARNERS = (DQN, DQN_QAT, DQN_QAT4, DQN_QAT_W, DQN_QAT_W4)


class TinyEnv:
    """Small deterministic rollout for checking consecutive learn() calls."""

    def __init__(self):
        self.observation_space = gym.spaces.Box(-1, 1, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, seed=None):
        self.action_space.seed(seed)
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(8, dtype=np.float32), 0.0, False, False, {}


def create(cls, **kwargs):
    env = TinyEnv()
    return cls("MlpPolicy", env, policy_kwargs={"net_arch": [4, 4]},
               learning_starts=1_000_000, verbose=0, **kwargs)


class ExplorationScheduleTests(unittest.TestCase):
    def test_all_learners_use_absolute_horizon_for_both_floors(self):
        for cls in LEARNERS:
            for floor in (0.05, 0.01):
                with self.subTest(cls=cls.__module__, floor=floor):
                    model = create(cls, exploration_final_eps=floor,
                                   exploration_decay_steps=EXPLORATION_DECAY_STEPS)
                    for step in (0, 5_000, 25_000, 250_000, 500_000, 750_000):
                        model.num_timesteps = step
                        expected = 1.0 + min(step / 500_000, 1.0) * (floor - 1.0)
                        for call_budget in (25_000, 500_000, 2_500_000):
                            self.assertAlmostEqual(model._exploration_rate(call_budget), expected)

    def test_additional_training_starts_at_point_four(self):
        for cls in LEARNERS:
            with self.subTest(cls=cls.__module__):
                model = create(cls, exploration_initial_eps=0.4,
                               exploration_final_eps=0.01, exploration_decay_steps=500_000)
                for step, expected in ((0, 0.4), (250_000, 0.205), (500_000, 0.01),
                                       (750_000, 0.01)):
                    model.num_timesteps = step
                    self.assertAlmostEqual(model._exploration_rate(25_000), expected)
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "refinement"
                    model.save(path)
                    restored = cls.load(path, env=TinyEnv())
                    self.assertEqual(restored.exploration_initial_eps, 0.4)
                    restored.num_timesteps = 250_000
                    self.assertAlmostEqual(restored._exploration_rate(25_000), 0.205)

    def test_consecutive_training_calls_do_not_restart_or_shorten_decay(self):
        for cls in LEARNERS:
            with self.subTest(cls=cls.__module__):
                model = create(cls, exploration_decay_steps=32)
                model.learn(total_timesteps=16, reset_num_timesteps=False)
                self.assertEqual(model.num_timesteps, 16)
                self.assertAlmostEqual(model._exploration_rate(16), 0.525)
                model.learn(total_timesteps=16, reset_num_timesteps=False)
                self.assertEqual(model.num_timesteps, 32)
                self.assertAlmostEqual(model._exploration_rate(16), 0.05)
                model.learn(total_timesteps=1, reset_num_timesteps=True)
                self.assertAlmostEqual(model._exploration_rate(16), 1 - 0.95 / 32)

    def test_saved_schedule_survives_loading_and_old_checkpoints_still_load(self):
        with tempfile.TemporaryDirectory() as temp:
            for index, cls in enumerate(LEARNERS):
                with self.subTest(cls=cls.__module__):
                    path = Path(temp) / str(index)
                    model = create(cls, exploration_decay_steps=500_000)
                    model.num_timesteps = 250_000
                    model.save(path)
                    loaded = cls.load(path, env=TinyEnv())
                    self.assertEqual(loaded.exploration_decay_steps, 500_000)
                    self.assertAlmostEqual(loaded._exploration_rate(25_000), 0.525)
                    checkpoint_path = Path(str(path) + ".zip")
                    checkpoint = torch.load(checkpoint_path, weights_only=True)
                    checkpoint.pop("exploration_decay_steps")
                    checkpoint.pop("exploration_initial_eps", None)
                    torch.save(checkpoint, checkpoint_path)
                    legacy = cls.load(path, env=TinyEnv())
                    self.assertIsNone(legacy.exploration_decay_steps)
                    self.assertAlmostEqual(legacy._exploration_rate(25_000), 0.05)

    def test_default_fraction_schedule_is_unchanged(self):
        for cls in LEARNERS:
            with self.subTest(cls=cls.__module__):
                model = create(cls, exploration_fraction=0.2)
                model.num_timesteps = 2_500
                self.assertAlmostEqual(model._exploration_rate(25_000), 0.525)
                model.num_timesteps = 5_000
                self.assertAlmostEqual(model._exploration_rate(25_000), 0.05)

    def test_pendulum_initial_epsilon_and_recovery_boost_are_unchanged(self):
        model = create(DQN_QAT_W, exploration_initial_eps=0.7,
                       exploration_final_eps=0.1, exploration_fraction=0.5)
        model.num_timesteps = 125_000
        self.assertAlmostEqual(model._exploration_rate(500_000), 0.4)
        model.training_guard_state = {"boost_start": 125_000, "boost_end": 150_000,
                                      "boost_epsilon": 0.6}
        self.assertAlmostEqual(model._exploration_rate(500_000), 0.6)

    def test_invalid_explicit_horizon_is_rejected(self):
        for cls in LEARNERS:
            for invalid in (0, -1):
                with self.subTest(cls=cls.__module__, horizon=invalid):
                    with self.assertRaises(ValueError):
                        create(cls, exploration_decay_steps=invalid)

    def test_additional_trainers_pass_refinement_start_epsilon(self):
        root = Path(__file__).resolve().parents[1] / "LunarLander" / "training"
        for name in ("train_base_extended", "train_qat8", "train_qat4",
                     "train_qat_weights8", "train_qat_weights4"):
            with self.subTest(trainer=name):
                tree = ast.parse((root / (name + ".py")).read_text())
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name)
                         and node.func.id in {"DQN", "DQN_QAT", "DQN_QAT_4", "DQN_QAT_W", "DQN_QAT_W_4"}]
                self.assertEqual(len(calls), 1)
                keywords = {arg.arg: arg.value for arg in calls[0].keywords}
                self.assertEqual(keywords["exploration_initial_eps"].id,
                                 "ADDITIONAL_EXPLORATION_INITIAL_EPS")

    def test_every_lunarlander_constructor_passes_shared_horizon(self):
        root = Path(__file__).resolve().parents[1] / "LunarLander" / "training"
        files = ("train_base_initial.py", "train_base_extended.py", "train_qat8.py",
                 "train_qat4.py", "train_qat_weights8.py", "train_qat_weights4.py",
                 "train_qat_weights_scratch.py")
        for filename in files:
            with self.subTest(filename=filename):
                tree = ast.parse((root / filename).read_text())
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name)
                         and node.func.id in {"DQN", "DQN_QAT", "DQN_QAT_W", "DQN_QAT_4", "DQN_QAT_W_4"}]
                self.assertEqual(len(calls), 1)
                keywords = {arg.arg: arg.value for arg in calls[0].keywords}
                self.assertEqual(keywords["exploration_decay_steps"].id, "EXPLORATION_DECAY_STEPS")


if __name__ == "__main__":
    unittest.main()
