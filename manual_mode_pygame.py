import os
import yaml
import numpy as np
import pygame

from core.env import AsteroidDefenseEnv
from core.visual_pygame import PygameRenderer


def _load_env(cfg_path=os.path.join("configs", "config.yaml")):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return AsteroidDefenseEnv(cfg["env"])


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


def run_manual(cfg_path=os.path.join("configs", "config_aim.yaml")):
    env = _load_env(cfg_path)

    renderer = PygameRenderer(title="Asteroid Defense - Manual")
    total_reward = 0.0
    frozen = False
    vel_buffer = {}

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

        renderer.draw(env, reward=reward, total_reward=total_reward)

        if done:
            if seed_list:
                obs, _ = env.reset(seed=seed_list[seed_idx % len(seed_list)])
                seed_idx += 1
            else:
                obs, _ = env.reset()
            total_reward = 0.0

    renderer.close()


if __name__ == "__main__":
    run_manual(cfg_path='./configs/config_eval.yaml')
