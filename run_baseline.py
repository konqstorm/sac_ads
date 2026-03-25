import numpy as np

from core.env import AsteroidDefenseEnv
from core.baseline import BaselineController
from visuals.gif_recorder import GIFRecorder
from core.runtime_options import (
    load_config,
    load_ursina_loop,
    resolve_do_gif,
    resolve_fps,
    resolve_gif_directory,
    resolve_gif_fps,
    resolve_gif_name,
    resolve_renderer,
)
from visuals.visual_pygame import PygameRenderer


def _load_env(cfg):
    return AsteroidDefenseEnv(cfg["env"])


def _seed_list(cfg):
    visual_cfg = cfg.get("visual", {})
    seeds = cfg.get("visual_seeds", visual_cfg.get("seeds"))
    return list(seeds) if seeds else []


def run_baseline(cfg_path="configs/config_eval.yaml", renderer=None):
    cfg = load_config(cfg_path)
    env = _load_env(cfg)
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

    seeds = _seed_list(cfg)
    seed_idx = 0

    def reset_env():
        nonlocal seed_idx, controller
        controller = BaselineController(env)
        if seeds:
            obs, _ = env.reset(seed=seeds[seed_idx % len(seeds)])
            seed_idx += 1
            return obs
        obs, _ = env.reset()
        return obs

    obs = reset_env()

    if renderer_name == "3d":
        run_ursina_loop = load_ursina_loop()
        run_ursina_loop(
            env=env,
            title="Asteroid Defense - Baseline (3D)",
            fps=fps,
            initial_obs=obs,
            on_episode_reset=reset_env,
            action_fn=lambda current_obs: controller.act(),
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
            obs = reset_env()
            total_reward = 0.0

    renderer.close()


if __name__ == "__main__":
    run_baseline()
