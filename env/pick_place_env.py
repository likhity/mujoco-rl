import os
import gymnasium as gym
import numpy as np
from mujoco import MjModel, MjData
from mujoco import mj_step, mj_resetDataKeyframe
import mujoco.viewer


class PickPlaceEnv(gym.Env):
    def __init__(self, reward_type="dense", distance_threshold=0.05, max_steps=100):
        xml_path = os.path.join(os.path.dirname(__file__), "../mujoco_assets/pick_place.xml")
        self.model = MjModel.from_xml_path(xml_path)
        self.data = MjData(self.model)

        self.viewer = None

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.max_steps = max_steps
        self.current_step = 0  # tracks step count per episode

        # Action: 2 joint torques
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation: joint angles (2) + object pos (3) + goal pos (3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mj_resetDataKeyframe(self.model, self.data, 0)
        self.current_step = 0

        # Reset arm joints
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        
        self.data.qpos[2] = 0.3 + 0.05 * np.random.randn()  # x of object
        self.data.qpos[3] = 0.2 + 0.05 * np.random.randn()  # y of object

        for _ in range(5):
            mj_step(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        action = np.clip(action, -1, 1)
        self.data.ctrl[:] = action

        for _ in range(5):
            mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward(obs, action)

        object_pos = obs[2:5]
        goal_pos = obs[5:]
        distance = np.linalg.norm(object_pos - goal_pos)
        success = distance < self.distance_threshold
        
        # Determine if episode is over
        terminated = success
        truncated = self.current_step >= self.max_steps

        info = {"is_success": success}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        joint_angles = self.data.qpos[:2]
        object_pos = self.data.xpos[self.model.body('object').id]
        goal_pos = self.model.site('goal_site').pos
        return np.concatenate([joint_angles, object_pos, goal_pos])

    def _compute_reward(self, obs, action):
        object_pos = obs[2:5]
        goal_pos = obs[5:]

        grip_site_pos = self.data.site('grip_site').xpos

        # Distance from gripper to object (encourages contact)
        grip_obj_dist = np.linalg.norm(grip_site_pos - object_pos)

        # Distance from object to goal (encourages transport)
        obj_goal_dist = np.linalg.norm(object_pos - goal_pos)
        
        control_cost = np.linalg.norm(action) ** 2
              
        w_near = 0.5
        reward_dist_weight = 1.0
        reward_control_weight = 0.1

        reward = -w_near * grip_obj_dist - reward_dist_weight * obj_goal_dist - reward_control_weight * control_cost
        if obj_goal_dist < self.distance_threshold:
            reward += 10.0
    
        return reward

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
