import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CameraEnv(gym.Env):
    def __init__(self, config):
        self.w = config["width"]
        self.h = config["height"]
        self.max_steps = config["max_steps"]
        self.delay = config["delay"]
        self.dt = config["dt"]

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        # Observation: sin(theta), cos(theta), normalized distance (0..1)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(3 + self.delay * 2 + 2,), dtype=np.float32
        )

        self.last_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        self.step_count = 0

        # объект (птица)
        self.obj_pos = np.array([0.5, 0.5])
        self.obj_vel = np.random.randn(2) * 0.1

        # камера (центр)
        self.cam_pos = np.array([0.5, 0.5])

        # очередь задержки
        self.action_queue = [np.zeros(2) for _ in range(self.delay)]

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # задержка
        self.action_queue.append(action)
        real_action = self.action_queue.pop(0)

        # обновляем камеру
        self.cam_pos += real_action * 0.05
        self.cam_pos = np.clip(self.cam_pos, 0, 1)

        # динамика "птицы"
        noise = np.random.randn(2) * 0.02
        self.obj_vel += noise
        self.obj_vel *= 0.95  # трение

        self.obj_pos += self.obj_vel * self.dt

        # отражение от границ
        for i in range(2):
            if self.obj_pos[i] < 0 or self.obj_pos[i] > 1:
                self.obj_vel[i] *= -1
                self.obj_pos[i] = np.clip(self.obj_pos[i], 0, 1)

        # reward по текущему видимому состоянию (в координатах камеры)
        rel = self.obj_pos - self.cam_pos
        # Камера видит область [-0.5, 0.5] от своего центра
        if -0.5 <= rel[0] < 0.5 and -0.5 <= rel[1] < 0.5:
            dist = np.linalg.norm(rel)
            reward = -dist
        else:
            reward = -1.0 # объект потерян

        done = self.step_count >= self.max_steps
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        rel = self.obj_pos - self.cam_pos
        
        # Если точка в кадре (от -0.5 до 0.5 от центра)
        if -0.5 <= rel[0] < 0.5 and -0.5 <= rel[1] < 0.5:
            is_visible = 1.0
            rx, ry = rel[0], rel[1]
        else:
            # Объект не виден
            is_visible = 0.0
            rx, ry = 0.0, 0.0

        actions_in_flight = np.array(self.action_queue).flatten()
        velocity = self.obj_vel
        
        # Размер: 3 (rx, ry, vis) + delay*2 + 2 = 3 + 6 + 2 = 11
        obs = np.concatenate(([rx, ry, is_visible], actions_in_flight, velocity))
        return obs.astype(np.float32)
