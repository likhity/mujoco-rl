from env.pick_place_env import PickPlaceEnv
from stable_baselines3 import PPO
import os

# Create the environment
env = PickPlaceEnv()

# Define the PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_pick_place_tensorboard/",
    device="cpu",
)

# Train the model
model.learn(total_timesteps=1_000_000)

# Save the trained model
save_path = "models/ppo_pick_place"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save(save_path)

print("✅ Training complete. Model saved at:", save_path)