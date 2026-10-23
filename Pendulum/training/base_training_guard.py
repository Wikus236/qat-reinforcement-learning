"""Recovery rules for long pendulum base training runs."""

from Pendulum.paths import APP_ROOT

import math

import torch

from rl_models.qat_weights import DQN_QAT_W, ReplayBuffer


EXPLORATION_FLOOR = 0.10
RECOVERY_EXPLORATION = 0.30
RECOVERY_STEPS = 50_000
RECOVERY_BOOST_STEPS = 500_000
RECOVERY_COOLDOWN_STEPS = 500_000
MIN_RECOVERY_TRAINING_STEPS = 3_000_000
BAD_EVALUATIONS_BEFORE_RECOVERY = 6
STALE_STEPS_BEFORE_EXPLORATION = 500_000
MIN_LEARNING_RATE = 1e-5
BASE_LEARNING_RATE = 1e-4
GUARD_VERSION = 2


class BaseTrainingGuard:
    def __init__(self, model, best_path, eval_freq):
        self.best_path = best_path
        self.eval_freq = eval_freq
        self.state = getattr(model, "training_guard_state", {}).copy()
        self.state.setdefault("bad_evaluations", 0)
        self.state.setdefault("last_improvement_step", model.num_timesteps)
        self.state.setdefault("cooldown_until", 0)
        self.state.setdefault("recoveries", 0)
        if self.state.get("version", 1) < GUARD_VERSION:
            # Undo the old guard's premature LR reductions once on resume.
            if self.state["recoveries"]:
                model.learning_rate = BASE_LEARNING_RATE
                for group in model.optimizer.param_groups:
                    group["lr"] = BASE_LEARNING_RATE
                self.state["cooldown_until"] = model.num_timesteps + RECOVERY_COOLDOWN_STEPS
            self.state["bad_evaluations"] = 0
            self.state["last_improvement_step"] = model.num_timesteps
            self.state.pop("boost_start", None)
            self.state.pop("boost_end", None)
            self.state["version"] = GUARD_VERSION
        model.training_guard_state = self.state

    @staticmethod
    def performance_dropped(metrics, best):
        # Require a useful baseline and a material drop, not a small fluctuation.
        balance_drop = (
            best["longest_balance_seconds"] >= 1.0
            and metrics["longest_balance_seconds"] < 0.75 * best["longest_balance_seconds"]
            and metrics["balance_fraction"] < 0.75 * best["balance_fraction"]
        )
        capture_drop = (
            best["capture_rate"] >= 0.5
            and metrics["capture_rate"] < best["capture_rate"] - 0.20
        )
        rail_drop = metrics["rail_failure_rate"] > best["rail_failure_rate"] + 0.20
        return balance_drop or capture_drop or rail_drop

    def boost_exploration(self, model):
        self.state["boost_start"] = model.num_timesteps
        self.state["boost_end"] = model.num_timesteps + RECOVERY_BOOST_STEPS
        self.state["boost_epsilon"] = RECOVERY_EXPLORATION
        self.state["last_improvement_step"] = model.num_timesteps

    def recover(self, model):
        best = DQN_QAT_W.load(self.best_path, env=model.env)
        model.q_net.load_state_dict(best.q_net.state_dict())
        model.target_net.load_state_dict(best.q_net.state_dict())
        # Preserve the run's step counter; reset stale optimizer/buffer state.
        model.learning_rate = min(model.learning_rate, max(MIN_LEARNING_RATE, model.learning_rate * 0.5))
        model.optimizer = torch.optim.Adam(model.q_net.parameters(), lr=model.learning_rate)
        model.replay_buffer = ReplayBuffer(model.buffer_size)
        model.learning_starts = model.num_timesteps + RECOVERY_STEPS
        self.state["bad_evaluations"] = 0
        self.state["recoveries"] += 1
        self.state["cooldown_until"] = model.learning_starts + RECOVERY_COOLDOWN_STEPS
        self.boost_exploration(model)
        return "restored_best"

    @staticmethod
    def is_finite(model, metrics):
        finite_metrics = all(math.isfinite(metrics[key]) for key in (
            "mean_reward", "balance_fraction", "longest_balance_seconds", "capture_rate"
        ))
        finite_weights = all(torch.isfinite(p).all().item() for p in model.q_net.parameters())
        return finite_metrics and finite_weights

    def observe(self, model, metrics, best_metrics, improved):
        if not self.is_finite(model, metrics):
            return self.recover(model)
        if improved:
            self.state["bad_evaluations"] = 0
            self.state["last_improvement_step"] = model.num_timesteps
            return "new_best"

        if model.num_timesteps < self.state["cooldown_until"]:
            return "recovery_warmup" if model.num_timesteps <= model.learning_starts else "recovery_cooldown"

        # Partial balance during early learning is not a rollback baseline.
        reliable_baseline = (
            model.num_timesteps >= MIN_RECOVERY_TRAINING_STEPS
            and best_metrics["capture_rate"] >= 0.8
            and best_metrics["longest_balance_seconds"] >= 5.0
            and best_metrics["balance_fraction"] >= 0.25
        )
        if reliable_baseline and self.performance_dropped(metrics, best_metrics):
            self.state["bad_evaluations"] += 1
        else:
            self.state["bad_evaluations"] = 0
        if self.state["bad_evaluations"] >= BAD_EVALUATIONS_BEFORE_RECOVERY:
            return self.recover(model)

        if model.num_timesteps - self.state["last_improvement_step"] >= STALE_STEPS_BEFORE_EXPLORATION:
            self.boost_exploration(model)
            return "exploration_boost"
        return "evaluated"
