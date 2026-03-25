import numpy as np
import pygame

from core.env import AsteroidDefenseEnv
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


def _format_obs(obs):
    obs = np.asarray(obs, dtype=np.float32).flatten()
    slot_dim = 5
    slots_total = 5
    expected_len = slots_total * slot_dim + 4

    if len(obs) < expected_len:
        obs = np.pad(obs, (0, expected_len - len(obs)), constant_values=0.0)

    labels = []
    for i in range(slots_total):
        labels.extend([
            f"ast{i}_err_yaw",
            f"ast{i}_err_pitch",
            f"ast{i}_time_norm",
            f"ast{i}_dist_norm",
            f"ast{i}_t_impact_norm",
        ])

    labels.extend(["hp_norm", "remaining_norm", "yaw", "pitch"])
    return " | ".join([f"{l}={v:+.3f}" for l, v in zip(labels, obs[:expected_len])])


def run_manual(cfg_path="configs/config_eval.yaml", renderer=None):
    cfg = load_config(cfg_path)
    env = _load_env(cfg)
    renderer_name = resolve_renderer(cfg, renderer=renderer)
    fps = resolve_fps(cfg, default=30)
    gif_fps = resolve_gif_fps(cfg, default=fps)
    gif_recorder = GIFRecorder(
        enabled=resolve_do_gif(cfg, default=False),
        directory=resolve_gif_directory(cfg, default="tmp_gif"),
        name=resolve_gif_name(cfg, default="manual_run.gif"),
        fps=gif_fps,
    )

    seeds = _seed_list(cfg)
    seed_idx = 0

    def reset_env():
        nonlocal seed_idx
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
            title="Asteroid Defense - Manual (3D)",
            fps=fps,
            initial_obs=obs,
            on_episode_reset=reset_env,
            manual_controls=True,
            gif_recorder=gif_recorder,
        )
        return

    renderer = PygameRenderer(title="Asteroid Defense - Manual", gif_recorder=gif_recorder)
    total_reward = 0.0
    frozen = False
    vel_buffer = {}

    running = True
    while running:
        running = renderer.process_events()
        for event in renderer.last_events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                frozen = not frozen
                if frozen:
                    vel_buffer = {}
                    for a in env.asteroid_slots:
                        if a is not None:
                            vel_buffer[a["id"]] = a["vel"].copy()
                            a["vel"] = np.zeros_like(a["vel"])
                else:
                    for a in env.asteroid_slots:
                        if a is not None and a["id"] in vel_buffer:
                            a["vel"] = vel_buffer[a["id"]].copy()
                    vel_buffer = {}

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

        if frozen:
            for a in env.asteroid_slots:
                if a is not None:
                    if a["id"] not in vel_buffer:
                        vel_buffer[a["id"]] = a["vel"].copy()
                    a["vel"] = np.zeros_like(a["vel"])

        print(_format_obs(obs))
        print(f"reward={reward:+.3f}\n")

        renderer.draw(env, reward=reward, total_reward=total_reward, fps=fps)

        if done:
            obs = reset_env()
            total_reward = 0.0

    renderer.close()


if __name__ == "__main__":
    run_manual()
