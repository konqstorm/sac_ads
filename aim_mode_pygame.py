import os
import yaml
import numpy as np

from core.env import AsteroidDefenseEnv
from core.visual_pygame import PygameRenderer
from core.aimer import FrozenAimer
from core.aim_utils import extract_aim_obs


def _load_env(cfg_path=os.path.join("configs", "config_aim.yaml")):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return AsteroidDefenseEnv(cfg["env"]), cfg


def run_aim(cfg_path=os.path.join("configs", "config_aim.yaml")):
    env, cfg = _load_env(cfg_path)

    aim_cfg = cfg.get("aim", {})
    aimer_cfg = cfg.get("aimer", {})
    obs_dim = aim_cfg.get("obs_dim", 7)
    act_dim = aim_cfg.get("action_dim", 3)
    weights_dir = aimer_cfg.get("weights_dir", cfg.get("train", {}).get("save_dir", "results/weights_aim"))
    deterministic = bool(aimer_cfg.get("deterministic_eval", True))
    deterministic = False

    aimer = FrozenAimer(obs_dim, act_dim, weights_dir)

    renderer = PygameRenderer(title="Asteroid Defense - Aimer Only")
    total_reward = 0.0

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

        aim_obs = extract_aim_obs(obs, 0)
        action = aimer.act(aim_obs, deterministic=deterministic).astype(np.float32)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

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
    run_aim()
