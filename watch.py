from env.bowling_env import BowlingEnv
from stable_baselines3 import PPO
import time

import gymnasium as gym

# Load environment and trained model
env = BowlingEnv()
env.render_mode="human"
model = PPO.load("models/ppo_pusher", device="cpu")

# Reset environment
obs, _ = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    time.sleep(0.01)

    if terminated or truncated:
        obs, _ = env.reset()