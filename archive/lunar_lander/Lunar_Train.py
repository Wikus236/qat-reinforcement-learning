from LunarLander.paths import APP_ROOT
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


ENV_NAME = "LunarLander-v3"
SAVE_PATH = Path(f"{APP_ROOT}/models/initial_fp32/dqn_lander")

SAVE_PATH.parent.mkdir(exist_ok=True)


if __name__ == "__main__":
    env = gym.make(
        ENV_NAME,
        gravity=-10.0,
        enable_wind=False,
    )

    policy_kwargs = dict(
        net_arch=[256, 256]
    )

    if Path(str(SAVE_PATH) + ".zip").exists():
        model = DQN.load(SAVE_PATH, env=env)
        print("Loaded existing DQN model")
    else:
        model = DQN(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=1e-4,
            buffer_size=100_000, # Store the last 100,000 transitions in the replay buffer
            learning_starts=10_000, # Collect 10,000 transitions before starting training
            batch_size=64,
            gamma=0.99, # Discount factor
            train_freq=4, # Train the model every 4 steps
            gradient_steps=1, # Number of gradient steps per training iteration
            target_update_interval=1_000, # Update the target network every 1,000 steps
            exploration_fraction=0.2, # Fraction of entire training period over which the exploration rate decreases
            exploration_final_eps=0.05, # Final value of random action probability
            verbose=1,
        )
        print("Created new DQN model")

    model.learn(
        total_timesteps=1_500_000,
        log_interval=1,
        reset_num_timesteps=False,
    )

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=20,
        deterministic=True,
    )

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    model.save(SAVE_PATH)
    env.close()