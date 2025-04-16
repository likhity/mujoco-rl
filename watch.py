from env.pick_place_env import PickPlaceEnv
from stable_baselines3 import PPO
import time

# Load environment and trained model
env = PickPlaceEnv()
model = PPO.load("models/ppo_pick_place", device="cpu")

# Reset environment
obs, _ = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    time.sleep(0.01)

    if terminated or truncated:
        obs, _ = env.reset()