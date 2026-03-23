import numpy as np
import gymnasium as gym
from gymnasium import spaces

class AsteroidDefenseEnv(gym.Env):
    def __init__(self, config):
        self.max_steps = config.get("max_steps", 1000)
        self.dt = config.get("dt", 0.033)

        self.fov = config.get("fov", np.pi / 2.0)  # Углы поворота пушки
        self.max_ang_vel = config.get("max_ang_vel", 2.0)
        self.fire_threshold = config.get("fire_threshold", 0.0)

        self.projectile_speed = config.get("projectile_speed", 100.0)
        self.projectile_max_dist = config.get("projectile_max_dist", 150.0)

        self.asteroid_radius = config.get("asteroid_radius", 3.0)
        self.impact_radius = config.get("impact_radius", 5.0)
        self.max_asteroids = config.get("max_asteroids", 3)
        self.spawn_prob = config.get("spawn_prob", 0.05)
        
        # Размеры видимого "окна" в космос по осям X и Z (для нормировки)
        self.world_width = config.get("world_width", 100.0)
        self.world_height = config.get("world_height", 100.0)
        self.spawn_y = config.get("spawn_y", 100.0) # Дистанция спавна (ось Y)
        self.spawn_z_min = config.get("spawn_z_min", 20.0)
        self.spawn_z_max = config.get("spawn_z_max", 80.0)
        
        # Базовый угол возвышения пушки, чтобы по умолчанию она смотрела в центр окна
        self.base_pitch = config.get("base_pitch", np.pi / 4.0) # 45 градусов вверх

        self.asteroid_speed_y_min = config.get("asteroid_speed_y_min", 10.0)
        self.asteroid_speed_y_max = config.get("asteroid_speed_y_max", 30.0)
        self.asteroid_speed_xz_max = config.get("asteroid_speed_xz_max", 15.0)

        self.reward_hit = config.get("reward_hit", 1.0)
        self.reward_impact = config.get("reward_impact", -1.0)
        self.reward_shot = config.get("reward_shot", -0.1)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # 3 asteroids * (x, y, z, vx, vy, vz) + (yaw, pitch) = 20
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(20,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.yaw = 0.0
        self.pitch = 0.0
        self.asteroids = []
        self.projectiles = []

        initial = options.get("initial_asteroids") if options else min(3, self.max_asteroids)
        for _ in range(initial):
            self.asteroids.append(self._spawn_asteroid())

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Обновляем углы поворота пушки
        self.yaw += float(action[0]) * self.max_ang_vel * self.dt
        self.pitch += float(action[1]) * self.max_ang_vel * self.dt
        
        half_fov = self.fov / 2.0
        self.yaw = np.clip(self.yaw, -half_fov, half_fov)
        self.pitch = np.clip(self.pitch, -half_fov, half_fov)

        reward = 0.0

        if action[2] > self.fire_threshold:
            self.projectiles.append(self._spawn_projectile())
            reward += self.reward_shot

        # Обновляем снаряды
        new_projectiles = []
        for p in self.projectiles:
            p["pos"] = p["pos"] + p["vel"] * self.dt
            if np.linalg.norm(p["pos"]) <= self.projectile_max_dist:
                new_projectiles.append(p)
        self.projectiles = new_projectiles

        # Обновляем астероиды
        for a in self.asteroids:
            a["pos"] = a["pos"] + a["vel"] * self.dt

        # Коллизии: снаряд попал в астероид (честная 3D проверка)
        remaining_asteroids = []
        # Фикс: расширяем радиус для быстрых пуль, чтобы не пролетали сквозь астероид
        effective_radius = self.asteroid_radius + (self.projectile_speed * self.dt) / 2.0
        
        for a in self.asteroids:
            hit = False
            for p in self.projectiles:
                if np.linalg.norm(a["pos"] - p["pos"]) <= effective_radius:
                    hit = True
                    p["hit"] = True
                    reward += self.reward_hit
                    break
            if not hit:
                remaining_asteroids.append(a)

        self.asteroids = remaining_asteroids
        self.projectiles = [p for p in self.projectiles if not p.get("hit")]

        # Коллизии: астероид ударил базу (координата Y <= 0)
        survivors = []
        for a in self.asteroids:
            if a["pos"][1] <= 0.0 or np.linalg.norm(a["pos"]) <= self.impact_radius:
                reward += self.reward_impact
            else:
                survivors.append(a)
        self.asteroids = survivors

        # Спавн новых астероидов
        if len(self.asteroids) < self.max_asteroids:
            if np.random.rand() < self.spawn_prob:
                self.asteroids.append(self._spawn_asteroid())

        done = self.step_count >= self.max_steps
        return self._get_obs(), reward, done, False, {}

    def _spawn_asteroid(self):
        # Спавним строго "вдалеке" по оси Y, раскидывая по X и Z
        x = np.random.uniform(-self.world_width/2, self.world_width/2)
        y = self.spawn_y
        z = np.random.uniform(self.spawn_z_min, self.spawn_z_max)

        # Скорость направлена К игроку (-Y)
        vx = np.random.uniform(-self.asteroid_speed_xz_max, self.asteroid_speed_xz_max)
        vy = -np.random.uniform(self.asteroid_speed_y_min, self.asteroid_speed_y_max)
        vz = np.random.uniform(-self.asteroid_speed_xz_max, self.asteroid_speed_xz_max)

        return {
            "pos": np.array([x, y, z], dtype=np.float32),
            "vel": np.array([vx, vy, vz], dtype=np.float32),
        }

    def _gun_direction(self):
        # Перевод углов поворота в 3D направляющий вектор
        pitch_total = self.base_pitch + self.pitch
        x_dir = np.cos(pitch_total) * np.sin(self.yaw)
        y_dir = np.cos(pitch_total) * np.cos(self.yaw)
        z_dir = np.sin(pitch_total)
        return np.array([x_dir, y_dir, z_dir], dtype=np.float32)

    def _spawn_projectile(self):
        direction = self._gun_direction()
        return {
            "pos": np.zeros(3, dtype=np.float32), # Вылет из (0,0,0)
            "vel": direction * self.projectile_speed,
        }

    def _get_obs(self):
        obs = []
        # Сортируем астероиды по близости к игроку
        if self.asteroids:
            dists = [np.linalg.norm(a["pos"]) for a in self.asteroids]
            order = np.argsort(dists)
            selected = [self.asteroids[i] for i in order[:3]]
        else:
            selected = []

        for a in selected:
            # Нормализация для нейросети (от -1 до 1)
            nx = np.clip(a["pos"][0] / (self.world_width/2), -1.0, 1.0)
            ny = np.clip(a["pos"][1] / self.spawn_y, 0.0, 1.0) * 2.0 - 1.0
            nz = np.clip(a["pos"][2] / self.world_height, 0.0, 1.0) * 2.0 - 1.0
            
            nvx = np.clip(a["vel"][0] / self.asteroid_speed_xz_max, -1.0, 1.0)
            nvy = np.clip(a["vel"][1] / self.asteroid_speed_y_max, -1.0, 1.0)
            nvz = np.clip(a["vel"][2] / self.asteroid_speed_xz_max, -1.0, 1.0)

            obs.extend([nx, ny, nz, nvx, nvy, nvz])

        # Забиваем отсутствующие астероиды значениями "очень далеко"
        while len(obs) < 18:
            obs.extend([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        yaw_n = np.clip(self.yaw / (self.fov / 2.0), -1.0, 1.0)
        pitch_n = np.clip(self.pitch / (self.fov / 2.0), -1.0, 1.0)
        obs.extend([yaw_n, pitch_n])

        return np.array(obs, dtype=np.float32)