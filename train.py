from env.bowling_env import BowlingEnv
from stable_baselines3 import PPO
import os

import gymnasium as gym

# Create the environment
env = BowlingEnv()

# Define the PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_bowling_tensorboard/",
    device="cpu",
)

# Train the model
model.learn(total_timesteps=100_000)

# Save the trained model
save_path = "models/ppo_bowling"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save(save_path)

print("✅ Training complete. Model saved at:", save_path)