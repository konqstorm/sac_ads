import numpy as np
import pygame

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


def _format_obs(obs):
    labels = []
    for i in range(3):
        labels.extend([
            f"ast{i}_x",
            f"ast{i}_y",
            f"ast{i}_z",
            f"ast{i}_vx",
            f"ast{i}_vy",
            f"ast{i}_vz",
        ])
    labels.extend(["yaw", "pitch"])
    return " | ".join([f"{l}={v:+.3f}" for l, v in zip(labels, obs)])


def run_manual(cfg_path="config.yaml", renderer=None):
    cfg = load_config(cfg_path)
    env = _load_env(cfg)
    obs, _ = env.reset()
    renderer_name = resolve_renderer(cfg, renderer=renderer)
    fps = resolve_fps(cfg, default=30)
    gif_fps = resolve_gif_fps(cfg, default=fps)
    gif_recorder = GIFRecorder(
        enabled=resolve_do_gif(cfg, default=False),
        directory=resolve_gif_directory(cfg, default="tmp_gif"),
        name=resolve_gif_name(cfg, default="manual_run.gif"),
        fps=gif_fps,
    )

    if renderer_name == "3d":
        run_ursina_loop = load_ursina_loop()
        run_ursina_loop(
            env=env,
            title="Asteroid Defense - Manual (3D)",
            fps=fps,
            initial_obs=obs,
            manual_controls=True,
            extra_lines_fn=lambda current_obs: [_format_obs(current_obs)],
            gif_recorder=gif_recorder,
        )
        return

    renderer = PygameRenderer(title="Asteroid Defense - Manual", gif_recorder=gif_recorder)
    total_reward = 0.0

    running = True
    while running:
        running = renderer.process_events()
        keys = pygame.key.get_pressed()

        yaw_action, pitch_action, fire_action = 0.0, 0.0, -1.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            yaw_action -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            yaw_action += 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            pitch_action += 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            pitch_action -= 1
        if keys[pygame.K_SPACE]:
            fire_action = 1

        action = np.array([yaw_action, pitch_action, fire_action], dtype=np.float32)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        print(_format_obs(obs))
        print(f"reward={reward:+.3f}\n")

        renderer.draw(env, reward=reward, total_reward=total_reward, fps=fps)

        if done:
            obs, _ = env.reset()
            total_reward = 0.0

    renderer.close()


if __name__ == "__main__":
    run_manual()
