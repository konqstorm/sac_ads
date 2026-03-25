import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AsteroidDefenseEnv(gym.Env):
    def __init__(self, config):
        self.max_steps = config.get("max_steps", 1000)
        self.dt = config.get("dt", 0.033)

        self.fov = config.get("fov", np.pi / 2.0)
        self.max_ang_vel = config.get("max_ang_vel", 2.0)
        self.fire_threshold = config.get("fire_threshold", 0.5)

        self.projectile_speed = config.get("projectile_speed", 100.0)
        self.projectile_max_dist = config.get("projectile_max_dist", 150.0)

        self.asteroid_radius = config.get("asteroid_radius", 3.0)
        self.impact_radius = config.get("impact_radius", 5.0)
        self.max_asteroids = config.get("max_asteroids", 3)
        self.spawn_prob = config.get("spawn_prob", 0.05)
        self.total_asteroids = config.get("total_asteroids", 40)
        self.max_hp = config.get("max_hp", 20)

        self.world_width = config.get("world_width", 100.0)
        self.world_height = config.get("world_height", 100.0)
        self.spawn_y = config.get("spawn_y", 100.0)
        self.spawn_z_min = config.get("spawn_z_min", 20.0)
        self.spawn_z_max = config.get("spawn_z_max", 80.0)
        self.spawn_x_range = config.get("spawn_x_range")
        self.spawn_z_center = config.get("spawn_z_center")
        self.spawn_z_range = config.get("spawn_z_range")
        self.spawn_ring_min = config.get("spawn_ring_min")
        self.spawn_ring_max = config.get("spawn_ring_max")

        base_pitch = config.get("base_pitch")
        if base_pitch is None:
            if self.spawn_z_center is not None:
                target_z = float(self.spawn_z_center)
            else:
                target_z = float(self.spawn_z_min + self.spawn_z_max) / 2.0
            ratio = target_z / max(1e-6, float(self.projectile_max_dist))
            ratio = np.clip(ratio, -0.99, 0.99)
            base_pitch = float(np.arcsin(ratio))
        self.base_pitch = base_pitch

        self.asteroid_speed_y_min = config.get("asteroid_speed_y_min", 10.0)
        self.asteroid_speed_y_max = config.get("asteroid_speed_y_max", 30.0)
        self.asteroid_speed_xz_max = config.get("asteroid_speed_xz_max", 15.0)

        self.reward_hit = config.get("reward_hit", 1.0)
        self.reward_impact = config.get("reward_impact", -1.0)
        self.reward_shot = config.get("reward_shot", 0.0)
        self.reward_no_shot = config.get("reward_no_shot", 0.0)
        self.reward_win = config.get("reward_win", 0.0)
        self.reward_lose = config.get("reward_lose", 0.0)

        # NOTE: aim_reward убран намеренно.
        # В hybrid-режиме baseline наводится сам — агент отвечает только
        # за выбор цели и получает сигнал исключительно от hits/impacts.

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # obs: 5 астероидов * 5 признаков + hp_norm + remaining_norm + yaw + pitch = 29
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(29,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.yaw = 0.0
        self.pitch = 0.0
        self.asteroids = []
        self.asteroid_slots = [None for _ in range(self.max_asteroids)]
        self.projectiles = []
        self.kills = 0
        self.hull_damage = 0
        self._next_asteroid_id = 0
        self.asteroids_remaining = int(self.total_asteroids)
        self.hp = int(self.max_hp)

        if options and "initial_asteroids" in options:
            initial = int(options["initial_asteroids"])
        else:
            initial = min(self.max_asteroids, self.asteroids_remaining)
        for _ in range(initial):
            self._spawn_into_first_empty_slot()
            self.asteroids_remaining -= 1
        self._refresh_asteroids_list()

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        self.yaw += float(action[0]) * self.max_ang_vel * self.dt
        self.pitch += float(action[1]) * self.max_ang_vel * self.dt

        half_fov = self.fov / 2.0
        self.yaw = np.clip(self.yaw, -half_fov, half_fov)
        self.pitch = np.clip(self.pitch, -half_fov, half_fov)

        pitch_total = self.base_pitch + self.pitch
        if pitch_total < 0.0:
            self.pitch = -self.base_pitch
        elif pitch_total > np.pi / 2.0:
            self.pitch = (np.pi / 2.0) - self.base_pitch

        reward = 0.0

        if action[2] > self.fire_threshold:
            self.projectiles.append(self._spawn_projectile())
            reward += self.reward_shot
        else:
            reward += self.reward_no_shot

        new_projectiles = []
        for p in self.projectiles:
            p["pos"] = p["pos"] + p["vel"] * self.dt
            if np.linalg.norm(p["pos"]) <= self.projectile_max_dist:
                new_projectiles.append(p)
        self.projectiles = new_projectiles

        for i, a in enumerate(self.asteroid_slots):
            if a is None:
                continue
            a["pos"] = a["pos"] + a["vel"] * self.dt
            self.asteroid_slots[i] = a

        effective_radius = self.asteroid_radius + (self.projectile_speed * self.dt) / 2.0

        for i, a in enumerate(self.asteroid_slots):
            if a is None:
                continue
            hit = False
            for p in self.projectiles:
                if np.linalg.norm(a["pos"] - p["pos"]) <= effective_radius:
                    hit = True
                    p["hit"] = True
                    reward += self.reward_hit
                    self.kills += 1
                    break
            if hit:
                self.asteroid_slots[i] = None
        self.projectiles = [p for p in self.projectiles if not p.get("hit")]

        for i, a in enumerate(self.asteroid_slots):
            if a is None:
                continue
            if a["pos"][1] <= 0.0 or np.linalg.norm(a["pos"]) <= self.impact_radius:
                reward += self.reward_impact
                self.hull_damage += 1
                self.hp -= 1
                self.asteroid_slots[i] = None

        # Спавн из конечного пула — сразу заполняем до max_asteroids
        while self._num_active_asteroids() < self.max_asteroids and self.asteroids_remaining > 0:
            self._spawn_into_first_empty_slot()
            self.asteroids_remaining -= 1

        self._refresh_asteroids_list()

        obs = self._get_obs()

        done = False
        if self.hp <= 0:
            reward += self.reward_lose
            done = True
        elif self.asteroids_remaining == 0 and len(self.asteroids) == 0:
            reward += self.reward_win
            done = True
        elif self.step_count >= self.max_steps:
            done = True

        return obs, reward, done, False, {}

    # ------------------------------------------------------------------

    def _refresh_asteroids_list(self):
        self.asteroids = [a for a in self.asteroid_slots if a is not None]

    def _num_active_asteroids(self):
        return sum(1 for a in self.asteroid_slots if a is not None)

    def _spawn_into_first_empty_slot(self):
        for i in range(self.max_asteroids):
            if self.asteroid_slots[i] is None:
                self.asteroid_slots[i] = self._spawn_asteroid()
                return True
        return False

    def _spawn_asteroid(self):
        if (self.spawn_x_range is not None
                or self.spawn_z_center is not None
                or self.spawn_z_range is not None
                or self.spawn_ring_min is not None
                or self.spawn_ring_max is not None):
            x_range = self.spawn_x_range if self.spawn_x_range is not None else self.world_width / 6.0
            z_center = self.spawn_z_center if self.spawn_z_center is not None else self.world_height / 2.0
            z_range = self.spawn_z_range if self.spawn_z_range is not None else self.world_height / 6.0

            if self.spawn_ring_min is not None or self.spawn_ring_max is not None:
                r_min = self.spawn_ring_min if self.spawn_ring_min is not None else 0.0
                r_max = self.spawn_ring_max if self.spawn_ring_max is not None else max(x_range, z_range)
                r_max = max(r_max, r_min + 1e-3)
                theta = np.random.uniform(0.0, 2.0 * np.pi)
                r = np.random.uniform(r_min, r_max)
                x = r * np.cos(theta)
                z = z_center + r * np.sin(theta)
            else:
                x = np.random.uniform(-x_range, x_range)
                z = np.random.uniform(z_center - z_range, z_center + z_range)
                z = np.clip(z, 0.0, self.world_height)
        else:
            x = np.random.uniform(-self.world_width / 2, self.world_width / 2)
            z = np.random.uniform(self.spawn_z_min, self.spawn_z_max)
        y = self.spawn_y

        vx = np.random.uniform(-self.asteroid_speed_xz_max, self.asteroid_speed_xz_max)
        vy = -np.random.uniform(self.asteroid_speed_y_min, self.asteroid_speed_y_max)
        vz = np.random.uniform(-self.asteroid_speed_xz_max, self.asteroid_speed_xz_max)

        ast = {
            "id": self._next_asteroid_id,
            "pos": np.array([x, y, z], dtype=np.float32),
            "vel": np.array([vx, vy, vz], dtype=np.float32),
        }
        self._next_asteroid_id += 1
        return ast

    def _gun_direction(self):
        pitch_total = self.base_pitch + self.pitch
        x_dir = np.cos(pitch_total) * np.sin(self.yaw)
        y_dir = np.cos(pitch_total) * np.cos(self.yaw)
        z_dir = np.sin(pitch_total)
        return np.array([x_dir, y_dir, z_dir], dtype=np.float32)

    def _solve_intercept(self, r, v, s):
        a = np.dot(v, v) - s * s
        b = 2.0 * np.dot(r, v)
        c = np.dot(r, r)
        if abs(a) < 1e-6:
            if abs(b) < 1e-6:
                return None
            t = -c / b
            return t if t > 1e-4 else None
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        candidates = [t for t in (t1, t2) if t > 1e-4]
        return min(candidates) if candidates else None

    def _direction_to_yaw_pitch(self, direction):
        d = direction / (np.linalg.norm(direction) + 1e-8)
        pitch_total = np.arcsin(np.clip(d[2], -1.0, 1.0))
        yaw = np.arctan2(d[0], d[1])
        pitch = pitch_total - self.base_pitch
        return yaw, pitch

    def _spawn_projectile(self):
        direction = self._gun_direction()
        return {
            "pos": np.zeros(3, dtype=np.float32),
            "vel": direction * self.projectile_speed,
        }

    def _get_obs(self):
        obs = []

        slots = self.asteroid_slots[:5]

        half_fov = self.fov / 2.0
        max_time = self.spawn_y / max(1e-6, self.asteroid_speed_y_min)
        max_dist = np.linalg.norm(
            np.array([self.world_width / 2.0, self.spawn_y, self.world_height], dtype=np.float32)
        )

        for a in slots:
            if a is None:
                obs.extend([0.0, 0.0, 1.0, 1.0, 1.0])
                continue
            r = a["pos"].astype(np.float32)
            v = a["vel"].astype(np.float32)
            t = self._solve_intercept(r, v, self.projectile_speed)
            aim_point = r + v * t if t is not None else r
            target_yaw, target_pitch = self._direction_to_yaw_pitch(aim_point)

            err_yaw = target_yaw - self.yaw
            err_pitch = target_pitch - self.pitch

            err_yaw_n = np.clip(err_yaw / half_fov, -1.0, 1.0)
            err_pitch_n = np.clip(err_pitch / half_fov, -1.0, 1.0)
            time_norm = 1.0 if t is None else np.clip(t / max_time, 0.0, 1.0)
            dist_norm = np.clip(np.linalg.norm(r) / max_dist, 0.0, 1.0)

            t_impact = abs(a["pos"][1] / min(-1e-3, a["vel"][1]))
            t_impact_norm = np.clip(t_impact / max_time, 0.0, 1.0)
            obs.extend([err_yaw_n, err_pitch_n, time_norm, dist_norm, t_impact_norm])

        # Фикс: 5 слотов по 5 признаков = 25
        while len(obs) < 25:
            obs.extend([0.0, 0.0, 1.0, 1.0, 1.0])

        # Глобальное состояние: агент должен знать насколько критична ситуация
        hp_norm = np.clip(self.hp / max(1, self.max_hp), 0.0, 1.0)
        remaining_norm = np.clip(
            self.asteroids_remaining / max(1, self.total_asteroids), 0.0, 1.0
        )
        obs.extend([hp_norm, remaining_norm])

        # текущий угол пушки (нормализованный)
        yaw_n = np.clip(self.yaw / half_fov, -1.0, 1.0)
        pitch_n = np.clip(self.pitch / half_fov, -1.0, 1.0)
        obs.extend([yaw_n, pitch_n])

        return np.array(obs, dtype=np.float32)
