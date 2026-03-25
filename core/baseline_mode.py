import os
import yaml
import numpy as np

from core.env import AsteroidDefenseEnv
from core.visual_pygame import PygameRenderer


def _load_env(cfg_path=os.path.join("configs", "config.yaml")):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return AsteroidDefenseEnv(cfg["env"])


def _solve_intercept(r, v, s):
    # Solve |r + v t| = s t
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


def _direction_to_yaw_pitch(direction, base_pitch):
    d = direction / (np.linalg.norm(direction) + 1e-8)
    pitch_total = np.arcsin(np.clip(d[2], -1.0, 1.0))
    yaw = np.arctan2(d[0], d[1])
    pitch = pitch_total - base_pitch
    return yaw, pitch


class BaselineController:
    def __init__(self, env):
        self.env = env
        self.fired_ids = set()
        self.target_id = None

    def _select_target(self):
        candidates = [a for a in self.env.asteroids if a["id"] not in self.fired_ids]
        if not candidates:
            return None
        dists = [np.linalg.norm(a["pos"]) for a in candidates]
        return candidates[int(np.argmin(dists))]

    def _aim_point(self, asteroid):
        r = asteroid["pos"].astype(np.float32)
        v = asteroid["vel"].astype(np.float32)
        t = _solve_intercept(r, v, self.env.projectile_speed)
        if t is None:
            return r
        return r + v * t

    def act(self):
        target = self._select_target()
        if target is None:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)

        aim_point = self._aim_point(target)
        direction = aim_point
        if np.linalg.norm(direction) < 1e-6:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)

        target_yaw, target_pitch = _direction_to_yaw_pitch(direction, self.env.base_pitch)

        # Clamp target to FOV
        half_fov = self.env.fov / 2.0
        target_yaw = np.clip(target_yaw, -half_fov, half_fov)
        target_pitch = np.clip(target_pitch, -half_fov, half_fov)

        err_yaw = target_yaw - self.env.yaw
        err_pitch = target_pitch - self.env.pitch

        step = self.env.max_ang_vel * self.env.dt
        yaw_action = np.clip(err_yaw / step, -1.0, 1.0)
        pitch_action = np.clip(err_pitch / step, -1.0, 1.0)

        fire_action = -1.0
        if abs(err_yaw) < 0.02 and abs(err_pitch) < 0.02:
            if target["id"] not in self.fired_ids:
                fire_action = 1.0
                self.fired_ids.add(target["id"])

        return np.array([yaw_action, pitch_action, fire_action], dtype=np.float32)


def run_baseline(cfg_path=os.path.join("configs", "config.yaml")):
    env = _load_env(cfg_path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    visual_cfg = cfg.get("visual", {})
    seed_list = cfg.get("visual_seeds", visual_cfg.get("seeds"))
    seed_list = list(seed_list) if seed_list else []
    seed_idx = 0

    if seed_list:
        obs, _ = env.reset(seed=seed_list[seed_idx % len(seed_list)])
        seed_idx += 1
    else:
        obs, _ = env.reset()
    controller = BaselineController(env)

    renderer = PygameRenderer(title="Asteroid Defense - Baseline")
    total_reward = 0.0

    running = True
    while running:
        running = renderer.process_events()
        action = controller.act()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        renderer.draw(env, reward=reward, total_reward=total_reward)

        if done:
            if seed_list:
                obs, _ = env.reset(seed=seed_list[seed_idx % len(seed_list)])
                seed_idx += 1
            else:
                obs, _ = env.reset()
            controller = BaselineController(env)
            total_reward = 0.0

    renderer.close()


if __name__ == "__main__":
    run_baseline()
