from env.bowling_env import BowlingEnv
from stable_baselines3 import PPO
import numpy as np
# Load trained model
model = PPO.load("models/ppo_bowling")
# Create environment
env = BowlingEnv()
n_episodes = 100
episode_rewards = []
successes = []
episode_lengths = []
for ep in range(n_episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated
    episode_rewards.append(total_reward)
    successes.append(info["is_success"])
    episode_lengths.append(step_count)
    print(f"Episode {ep+1} | Reward: {total_reward:.2f} | Success: {info['is_success']} | Steps: {step_count} ")
# Summary
print("\n📊 Evaluation Summary:")
print(f"✅ Success rate: {np.mean(successes) * 100:.1f}%")
print(f"🏆 Avg reward: {np.mean(episode_rewards):.2f}")
print(f"⏱️ Avg episode length: {np.mean(episode_lengths):.1f} steps")
