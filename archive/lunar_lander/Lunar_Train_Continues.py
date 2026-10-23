from LunarLander.paths import APP_ROOT
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy




ENV_NAME = "LunarLander-v3"
SAVE_PATH = Path(f"{APP_ROOT}/models/initial_fp32/ppo_lander")

SAVE_PATH.parent.mkdir(exist_ok=True)


if __name__ == "__main__":

    env = gym.make(ENV_NAME, continuous=True, gravity=-10.0,
                enable_wind=False, wind_power=15.0, turbulence_power=1.5)


    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )


    if Path(str(SAVE_PATH) + ".zip").exists():
        model = PPO.load(SAVE_PATH, env=env)
        print("Loaded existing model")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
        )
        print("Created new model")

    model.learn(total_timesteps=3_000_000,log_interval=1,reset_num_timesteps=False)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=20,
        deterministic=True,
    )

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    model.save(SAVE_PATH)
    env.close()