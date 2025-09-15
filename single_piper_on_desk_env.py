"""
Clean MuJoCo environment for Piper robot arm apple grasping task.
This environment uses only state observations (joint angles + end effector pose + apple position).
"""
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
import os
from scipy.spatial.transform import Rotation as R
import time


class PiperEnv(gym.Env):
    """
    Piper robot arm environment for apple grasping task using delta (incremental) actions.
    
    The neural network outputs delta joint angles [δq1, δq2, δq3, δq4, δq5, δq6, δq7] 
    representing incremental changes to joint positions rather than absolute positions.
    This approach provides smoother control and better learning stability.
    """
    def __init__(self, render_mode=None):
        super(PiperEnv, self).__init__()
        script_dir = os.path.dirname(os.path.realpath(__file__)) 
        xml_path = os.path.join(script_dir, 'model_assets', 'piper_on_desk', 'scene.xml')

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link6')

        # Support both old and new render API
        if render_mode is None:
            self.render_mode = False  # No rendering by default
        else:
            self.render_mode = render_mode in ["human", "rgb_array"]
        
        if self.render_mode:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)  # 创建一个被动渲染窗口(GUI)，可以实时查看仿真过程
            self.handle.cam.distance = 3  # 相机与目标的距离为 3
            self.handle.cam.azimuth = 0  # 方位角为 0 度
            self.handle.cam.elevation = -30  # 仰角为 -30 度
        else:
            self.handle = None

        # Joint limits for 7 DOF (6 joints + gripper)
        self.joint_limits = np.array([
            (-2.618, 2.618),
            (0, 3.14),
            (-2.697, 0),
            (-1.832, 1.832),
            (-1.22, 1.22),
            (-3.14, 3.14),
            (0, 0.035),
        ])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,))
        
        # Add camera parameters
        self.camera_width = 128
        self.camera_height = 128
        
        # Mixed observation space: dict with cameras and state
        self.observation_space = spaces.Dict({
            'rgb': spaces.Box(low=0, high=255, shape=(self.camera_height, self.camera_width, 3), dtype=np.uint8),
            'wrist_cam': spaces.Box(low=0, high=255, shape=(self.camera_height, self.camera_width, 3), dtype=np.uint8),
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        })

        self.workspace_limits = {
            'x': (0.1, 0.7),
            'y': (-0.7, 0.7),
            'z': (0.1, 0.7)
        }

        self.goal_reached = False
        self._reset_noise_scale = 0.0
        self.episode_len = 80
        
        self.init_qpos = np.zeros(8)
        self.init_qpos[1] = 1.1
        self.init_qpos[2] = -0.95
        self.init_qpos[4] = 0.976
        self.init_qpos[6] = 0.035
        self.init_qpos[7] = -0.035
        self.init_qvel = np.zeros(8)
        self.contact_streak = 0
        self.max_contact_streak = 30
        
        # Track previous gripper position for movement penalty
        self.prev_gripper_pos = 0.035   # Initialize to open position
        
        # Initialize persistent renderer for efficiency
        self._renderer = None
        # Always try to create renderer for RGB observations (needed even in headless mode)
        try:
            self._renderer = mujoco.Renderer(self.model, height=self.camera_height, width=self.camera_width)
        except Exception as e:
            print(f"Warning: Could not initialize renderer: {e}")
            self._renderer = None

    def _get_site_pos_ori(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"Site '{site_name}' not found")

        position = np.asarray(self.data.site(site_id).xpos, dtype=np.float32)
        xmat = np.asarray(self.data.site(site_id).xmat, dtype=np.float64)  # MuJoCo requires float64
        quaternion = np.zeros(4, dtype=np.float64)  # MuJoCo requires float64
        mujoco.mju_mat2Quat(quaternion, xmat)

        return position, quaternion.astype(np.float32)  # Convert back to float32
    
    def _get_body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found")
        
        position = np.asarray(self.data.body(body_id).xpos, dtype=np.float32)
        quaternion = np.asarray(self.data.body(body_id).xquat, dtype=np.float32)
        
        return position, quaternion

    def map_action_to_joint_deltas(self, action: np.ndarray) -> np.ndarray:
        """Map [-1, 1] action to joint angle increments with discrete gripper control."""
        max_delta_per_step = np.array([
            0.05, 0.03, 0.03, 0.03, 0.03, 0.05, 0.005
        ], dtype=np.float32)
        
        # Ensure action is a numpy array with proper dtype
        action = np.asarray(action, dtype=np.float32)
        
        # Process first 6 joints normally (continuous control)
        delta_action = action * max_delta_per_step
        
        # Process gripper action discretely (index 6)
        gripper_action = action[6]
        current_gripper_pos = self.data.qpos[6]  # Current gripper position
        
        if gripper_action < 0:
            # Close: set to 0
            delta_action[6] = 0.0 - current_gripper_pos
        else:
            # Open: set to 0.035  
            delta_action[6] = 0.035 - current_gripper_pos
        
        return delta_action
    
    def apply_joint_deltas_with_limits(self, current_qpos: np.ndarray, delta_action: np.ndarray) -> np.ndarray:
        """Apply delta action to current joint positions with limits."""
        # Ensure arrays are proper numpy arrays with consistent dtype
        current_qpos = np.asarray(current_qpos, dtype=np.float32)
        delta_action = np.asarray(delta_action, dtype=np.float32)
        
        new_qpos = current_qpos + delta_action
        lower_bounds = self.joint_limits[:, 0].astype(np.float32)
        upper_bounds = self.joint_limits[:, 1].astype(np.float32)
        return np.clip(new_qpos, lower_bounds, upper_bounds)
    
    def _set_state(self, qpos, qvel):
        assert qpos.shape == (8,) and qvel.shape == (8,)
        self.data.qpos[:8] = np.copy(qpos)
        self.data.qvel[:8] = np.copy(qvel)
        mujoco.mj_step(self.model, self.data)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=8
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=8
        )

        qpos[1] = 1.1
        qpos[2] = -0.95
        qpos[4] = 0.976
        qpos[6] = 0.035
        qpos[7] = -0.035

        self._set_state(qpos, qvel)
        self._reset_object_pose()
        
        obs = self._get_observation()
        
        self.step_number = 0
        self.goal_reached = False
        self.contact_streak = 0
        
        # Initialize previous gripper position for tracking movement
        self.prev_gripper_pos = self.data.qpos[6]   # Current gripper position

        return obs, {}  # Gymnasium API returns (observation, info)
    
    def set_goal_pose(self, goal_body_name, position, quat_wxyz):
        """Set target pose for a body."""
        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, goal_body_name)
        if goal_body_id == -1:
            raise ValueError(f"Body '{goal_body_name}' not found")

        goal_joint_id = self.model.body_jntadr[goal_body_id]
        goal_qposadr = self.model.jnt_qposadr[goal_joint_id]

        if goal_qposadr + 7 <= self.model.nq:
            self.data.qpos[goal_qposadr: goal_qposadr + 3] = position
            self.data.qpos[goal_qposadr + 3: goal_qposadr + 7] = quat_wxyz
    
    def _reset_object_pose(self):
        """Randomly place apple on table surface using arc-based positioning."""
        item_name = "apple"
        self.target_position, item_quat = self._get_body_pose(item_name)
        
        while True:
            # Arc-based positioning parameters (exact match with reference)
            center_x = -0.25
            center_y = 0
            radius = np.sqrt(0.28745)  # 0.53619m
            
            # Random angle on arc (-π/2 to π/2)
            theta = np.random.uniform(-np.pi / 2, np.pi / 2)
            
            # Base radius scaling
            rho = 0.8 * radius
            
            # Adjust radius based on angle to avoid certain regions
            if (theta < -np.pi / 9 and theta > -np.pi / 6) or (theta > np.pi / 9 and theta < np.pi / 6):
                rho = radius * np.random.uniform(0.45 / radius, 1)
            if (theta < np.pi / 9 and theta > -np.pi / 9):
                rho = radius * np.random.uniform(0.75, 1)
            
            # Calculate world position
            x_world_target = rho * np.cos(theta) + center_x
            y_world_target = rho * np.sin(theta) + center_y

            self.target_position[0] = x_world_target
            self.target_position[1] = y_world_target
            self.target_position[2] = 0.768
            
            # Check if apple would be placed too close to drop point (avoid overlap)
            try:
                drop_point_position, _ = self._get_site_pos_ori("drop_point")
                if ((abs(drop_point_position[0] - self.target_position[0]) < 0.1) and
                        (abs(drop_point_position[1] - self.target_position[1]) < 0.18)):
                    continue  # Too close to drop point, try again
                else:
                    break  # Good position found
            except:
                # If drop point not found, just use the position
                break
        
        self.set_goal_pose("apple", self.target_position, item_quat)
        
        # Reset apple velocity to zero
        apple_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "apple")
        if apple_body_id != -1:
            apple_joint_id = self.model.body_jntadr[apple_body_id]
            apple_qveladr = self.model.jnt_dofadr[apple_joint_id]
            
            # Reset linear and angular velocities to zero
            self.data.qvel[apple_qveladr:apple_qveladr + 6] = 0.0

    def _get_rgb_observation(self, camera_name):
        """Get RGB camera observation from specified camera."""
        # Get camera ID
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found")
        
        # Use persistent renderer for efficiency
        if self._renderer is None:
            try:
                self._renderer = mujoco.Renderer(self.model, height=self.camera_height, width=self.camera_width)
            except Exception as e:
                print(f"Warning: Could not create renderer: {e}")
                # Return dummy RGB array if rendering fails
                return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        
        try:
            self._renderer.update_scene(self.data, camera=camera_id)
            rgb_array = self._renderer.render()
            return rgb_array.astype(np.uint8)
        except Exception as e:
            print(f"Warning: Rendering failed: {e}")
            # Return dummy RGB array if rendering fails
            return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
    
    def _get_state_observation(self):
        """Get state observation: joint angles + end effector pose (no apple position)."""
        # Get joint states (7 joint angles)
        current_qpos = np.asarray(self.data.qpos[:7], dtype=np.float32)
        state_observation = np.concatenate([
            current_qpos,      # 7 joint angles
        ]).astype(np.float32)
        
        return state_observation

    def _get_observation(self):
        """Get mixed observation: RGB cameras + state."""
        rgb_obs = self._get_rgb_observation("3rd")
        wrist_cam_obs = self._get_rgb_observation("wrist_cam")
        state_obs = self._get_state_observation()
        
        return {
            'rgb': rgb_obs,
            'wrist_cam': wrist_cam_obs,
            'state': state_obs
        }

    def _check_contact_between_bodies(self, body1_name: str, body2_name: str) -> tuple[bool, float]:
        """Check contact between two bodies."""
        body1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
        body2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_name)
        
        if body1_id == -1 or body2_id == -1:
            return False, 0.0
            
        total_force = 0.0
        contact_found = False
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            geom1_body = self.model.geom_bodyid[geom1_id]
            geom2_body = self.model.geom_bodyid[geom2_id]
            
            if ((geom1_body == body1_id and geom2_body == body2_id) or 
                (geom1_body == body2_id and geom2_body == body1_id)):
                contact_found = True
                contact_force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, contact_force)
                force_magnitude = np.linalg.norm(contact_force[:3])
                total_force += force_magnitude
                
        return contact_found, total_force

    def _check_gripper_contact_with_table(self) -> bool:
        """Check if gripper contacts table."""
        contact_found = False
        
        for i in range(1, 9):
            link_name = f"link{i}"
            link_contact, _ = self._check_contact_between_bodies(link_name, "desk")
            if link_contact:
                contact_found = True
                break

        if not contact_found:
            wrist_contact, _ = self._check_contact_between_bodies("wrist_camera", "desk")
            if wrist_contact:
                contact_found = True

        return contact_found

    def _check_gripper_contact_with_apple(self) -> bool:
        """Check if gripper fingers (link7, link8) are in contact with apple."""
        contact_found = 0
        
        for link_name in ["link7", "link8"]:
            link_contact, _ = self._check_contact_between_bodies(link_name, "apple")
            if link_contact:
                contact_found += 1
        
        return contact_found == 2

    def _check_apple_fell_off_table(self) -> bool:
        """Check if apple fell off table."""
        apple_position, _ = self._get_body_pose('apple')
        x, y, z = apple_position
        
        table_x_min, table_x_max = -0.3, 0.3
        table_y_min, table_y_max = -0.6, 0.6
        table_surface_height = 0.74115
        
        if x < table_x_min or x > table_x_max:
            return True
        if y < table_y_min or y > table_y_max:
            return True
        if z < table_surface_height - 0.05:
            return True

        return False

    def _compute_reward(self):
        """Compute reward for apple grasping and dropping task."""
        # Get positions
        ee_position, _ = self._get_site_pos_ori("end_ee")
        apple_position, _ = self._get_body_pose('apple')
        drop_point_position, _ = self._get_site_pos_ori("drop_point")
        
        reward = 0.0
        current_gripper_pos = self.data.qpos[6]
        
        # Stage 1: Reaching reward (always available, smooth and substantial)
        tcp_to_obj_dist = np.linalg.norm(ee_position - apple_position)
        reaching_reward = 1.0 * (1 - np.tanh(3.0 * tcp_to_obj_dist))
        reward += reaching_reward

        # Stage 2: Close proximity bonus (when very close to apple)
        if tcp_to_obj_dist < 0.015:
            reward += 0.5
            
            # Stage 3: Gripper positioning reward (encourage closing when near apple)
            gripper_closure = np.clip(1.0 - (current_gripper_pos / 0.035), 0, 1)
            reward += 0.3 * gripper_closure
            
            # Stage 4: Grasping reward (significant but not overwhelming)
            is_grasping = self._check_gripper_contact_with_apple()
            if is_grasping:
                reward += 1.0
                
                # Small lifting reward to encourage keeping apple elevated
                TABLE_HEIGHT = 0.768
                current_height = apple_position[2]
                if current_height > TABLE_HEIGHT:
                    height_bonus = min(0.2, 0.2 * (current_height - TABLE_HEIGHT) / 0.1)  # Max 0.2 reward
                    reward += height_bonus
                
                # Stage 5: Direct drop approach reward
                apple_to_drop_dist = np.linalg.norm(apple_position - drop_point_position)
                drop_approach_reward = 10 * (1 - np.tanh(5.0 * apple_to_drop_dist))
                reward += drop_approach_reward
                
                # Stage 6: Final drop reward (when apple is near drop point)
                if apple_to_drop_dist < 0.03:  # 3cm threshold
                    reward += 1.0
                    # Reward for opening gripper near drop point
                    if current_gripper_pos > 0.025:  # Gripper opening threshold
                        print("Drop goal reached!")
                        reward += 2.0  # Final completion bonus
                        self.goal_reached = True
        
        # Apply penalties (but don't dominate the reward signal)
        if self._check_apple_fell_off_table():
            reward -= 2.0  # Reduced penalty
            self.goal_reached = False

        if self._check_gripper_contact_with_table():
            # print("Gripper is in contact with the table.")
            self.contact_streak += 1
            reward -= 3.0  # Reduced penalty
            if self.contact_streak > self.max_contact_streak:
                reward -= 10.0
        else:
            self.contact_streak = 0

        return reward

    def step(self, action):
        """Execute one environment step."""
        delta_action = self.map_action_to_joint_deltas(action)
        current_qpos = self.data.qpos[:7].copy()
        new_qpos = self.apply_joint_deltas_with_limits(current_qpos, delta_action)
        
        self.data.ctrl[:7] = new_qpos
        
        for i in range(15):
            mujoco.mj_step(self.model, self.data)
            
            # Render if viewer is available
            if self.render_mode and self.handle:
                self.handle.sync()
            
            current_qpos = self.data.qpos[:7].copy()
            pos_err = np.linalg.norm(new_qpos - current_qpos)

        self.step_number += 1
        observation = self._get_observation()
        reward = self._compute_reward()
        
        # Update previous gripper position for next step's penalty calculation
        self.prev_gripper_pos = self.data.qpos[6]   # Current gripper position

        apple_fell = self._check_apple_fell_off_table()
        contact_max = self.contact_streak > self.max_contact_streak
        terminated = self.goal_reached or apple_fell or contact_max

        truncated = self.step_number >= self.episode_len

        info = {
            'is_success': self.goal_reached,
            'total_reward': reward,
            'step_number': self.step_number,
            'goal_reached': self.goal_reached,
            'current_qpos': current_qpos.copy(),
            'delta_action': delta_action.copy(),
            'new_qpos': new_qpos.copy(),
        }

        return observation, reward, terminated, truncated, info  # Gymnasium API

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self):
        """Clean up resources"""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()


def make_env():
    """Factory function to create PiperEnv."""
    return PiperEnv(render_mode=None)


if __name__ == "__main__":
    # Simple test
    env = PiperEnv(render_mode="human")
    obs, info = env.reset()
    print(f"Observation type: {type(obs)}")
    print(f"Observation keys: {obs.keys()}")
    print(f"RGB shape: {obs['rgb'].shape}")
    print(f"Wrist cam shape: {obs['wrist_cam'].shape}")
    print(f"State shape: {obs['state'].shape}")
    print(f"Action space: {env.action_space}")
    
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
        if terminated or truncated:
            obs, info = env.reset()
        # time.sleep(0.1)