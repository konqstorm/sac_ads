import yaml
import numpy as np
import pygame

from env import AsteroidDefenseEnv
from visual_pygame import PygameRenderer


def _load_env(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
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


def run_manual(cfg_path="config.yaml"):
    env = _load_env(cfg_path)
    obs, _ = env.reset()

    renderer = PygameRenderer(title="Asteroid Defense - Manual")
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

        renderer.draw(env, reward=reward, total_reward=total_reward)

        if done:
            obs, _ = env.reset()
            total_reward = 0.0

    renderer.close()


if __name__ == "__main__":
    run_manual()
