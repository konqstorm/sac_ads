import numpy as np

from env import AsteroidDefenseEnv
from gif_recorder import GIFRecorder
from runtime_options import (
    load_config,
    load_ursina_loop,
    resolve_do_gif,
    resolve_fps,
    resolve_gif_directory,
    resolve_gif_fps,
    resolve_gif_name,
    resolve_renderer,
)
from visual_pygame import PygameRenderer


def _load_env(cfg):
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


def run_baseline(cfg_path="config.yaml", renderer=None):
    cfg = load_config(cfg_path)
    env = _load_env(cfg)
    obs, _ = env.reset()
    controller = BaselineController(env)
    renderer_name = resolve_renderer(cfg, renderer=renderer)
    fps = resolve_fps(cfg, default=30)
    gif_fps = resolve_gif_fps(cfg, default=fps)
    gif_recorder = GIFRecorder(
        enabled=resolve_do_gif(cfg, default=False),
        directory=resolve_gif_directory(cfg, default="tmp_gif"),
        name=resolve_gif_name(cfg, default="baseline_run.gif"),
        fps=gif_fps,
    )

    if renderer_name == "3d":
        run_ursina_loop = load_ursina_loop()
        state = {"controller": controller}

        def _act(_obs):
            return state["controller"].act()

        def _on_episode_reset():
            state["controller"] = BaselineController(env)

        run_ursina_loop(
            env=env,
            title="Asteroid Defense - Baseline (3D)",
            fps=fps,
            action_fn=_act,
            initial_obs=obs,
            on_episode_reset=_on_episode_reset,
            gif_recorder=gif_recorder,
        )
        return

    renderer = PygameRenderer(title="Asteroid Defense - Baseline", gif_recorder=gif_recorder)
    total_reward = 0.0

    running = True
    while running:
        running = renderer.process_events()
        action = controller.act()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        renderer.draw(env, reward=reward, total_reward=total_reward, fps=fps)

        if done:
            obs, _ = env.reset()
            controller = BaselineController(env)
            total_reward = 0.0

    renderer.close()


if __name__ == "__main__":
    run_baseline()
