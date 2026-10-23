"""Weight-only DQN refinement with retained successful base trajectories."""

from Pendulum.paths import APP_ROOT

import copy

import torch
import torch.nn.functional as F

from rl_models.qat_weights import DQN_QAT_W, ReplayBuffer


class MildDQN(DQN_QAT_W):
    def prepare_refinement(self, teacher, demo_env, seed, episodes=20):
        self.bootstrap_time_limits = True
        self.reference_net = copy.deepcopy(teacher.q_net).eval()
        for parameter in self.reference_net.parameters():
            parameter.requires_grad_(False)
        self.demo_buffer = ReplayBuffer(episodes * demo_env.max_episode_steps)
        # Old Adam moments belong to the base objective, not this refinement.
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.target_net.load_state_dict(self.q_net.state_dict())
        for episode in range(episodes):
            observation, _ = demo_env.reset(seed=seed + episode)
            while True:
                action, _ = teacher.predict(observation, deterministic=True)
                next_observation, reward, terminated, truncated, _ = demo_env.step(action)
                self.demo_buffer.add(observation.copy(), action, reward,
                                     next_observation.copy(), terminated)
                observation = next_observation
                if terminated or truncated:
                    break
        demo_env.close()

    def _train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        demo_count = self.batch_size // 4
        online = self.replay_buffer.sample(self.batch_size - demo_count, self.device)
        demo = self.demo_buffer.sample(demo_count, self.device)
        observations, actions, rewards, next_observations, terminals = (
            torch.cat((online_item, demo_item))
            for online_item, demo_item in zip(online, demo)
        )
        predictions = self.q_net(observations)
        chosen_values = predictions.gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            # Double DQN: select with the online net, evaluate with the target.
            next_actions = self.q_net(next_observations).argmax(1, keepdim=True)
            next_values = self.target_net(next_observations).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * (1.0 - terminals) * next_values
            reference = self.reference_net(observations)
            teacher_actions = reference.argmax(1, keepdim=True)
            teacher_values = reference.gather(1, teacher_actions)
            # Preserve the teacher's preference, rather than its old Q-value
            # scale (which changes when gamma/reward changes).
            margins = (teacher_values - reference).clamp(0.02, 0.2)
            margins.scatter_(1, teacher_actions, 0.0)
        policy_values = predictions.gather(1, teacher_actions)
        preservation_loss = (predictions + margins - policy_values).max(1).values.mean()
        loss = F.smooth_l1_loss(chosen_values, targets) + 5.0 * preservation_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        self.last_refinement_diagnostics = {
            "td_loss": float(F.smooth_l1_loss(chosen_values, targets).item()),
            "preservation_loss": float(preservation_loss.item()),
            "teacher_agreement": float((predictions.argmax(1) == teacher_actions[:, 0]).float().mean().item()),
        }
        return float(loss.item())
