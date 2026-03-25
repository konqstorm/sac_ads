import os
import yaml
import numpy as np
import torch
from core.env import AsteroidDefenseEnv
from core.models_discrete import Actor, Critic
from core.visual_pygame import PygameRenderer
from core.sac_discrete import SAC
from core.aimer import FrozenAimer
from core.two_stage_agent import TwoStageAgent


def _load_env(cfg_path=os.path.join("configs", "config_select.yaml")):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return AsteroidDefenseEnv(cfg["env"])


def _load_agent(env, weights_dir=None, cfg=None):
    obs_dim = env.observation_space.shape[0]
    if cfg is not None:
        act_dim = int(cfg.get("env", {}).get("max_asteroids", 5))
        agent_cfg = cfg.get("agent", {})
    else:
        act_dim = 5
        agent_cfg = {}
    actor = Actor(obs_dim, act_dim)
    critic1 = Critic(obs_dim, act_dim)
    critic2 = Critic(obs_dim, act_dim)
    agent = SAC(actor, critic1, critic2, agent_cfg)
    if weights_dir is None and cfg is not None:
        weights_dir = cfg.get("train", {}).get("save_dir_selector", "results/weights_selector")
    if weights_dir is None:
        weights_dir = "results/weights_selector"
    path = os.path.join(weights_dir, "actor.pt")
    actor.load_state_dict(torch.load(path, map_location="cpu"))
    # critics are optional for action, but load if present
    c1 = os.path.join(weights_dir, "critic1.pt")
    c2 = os.path.join(weights_dir, "critic2.pt")
    if os.path.exists(c1):
        critic1.load_state_dict(torch.load(c1, map_location="cpu"))
    if os.path.exists(c2):
        critic2.load_state_dict(torch.load(c2, map_location="cpu"))
    actor.eval()
    critic1.eval()
    critic2.eval()
    return agent

def run_agent(cfg_path=os.path.join("configs", "config_select.yaml"), weights_dir=None):
    env = _load_env(cfg_path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    agent = _load_agent(env, weights_dir=weights_dir, cfg=cfg)
    sample_every = cfg.get("agent", {}).get("target_sample_every", 5)

    aimer_cfg = cfg.get("aimer", {})
    aim_obs_dim = aimer_cfg.get("obs_dim", 7)
    aim_act_dim = aimer_cfg.get("action_dim", 3)
    aim_weights_dir = aimer_cfg.get("weights_dir", "results/weights_aim")
    aim_det = bool(aimer_cfg.get("deterministic_eval", True))
    aimer = FrozenAimer(aim_obs_dim, aim_act_dim, aim_weights_dir)
    commit_steps = cfg.get("agent", {}).get("commit_steps", 15)
    two_stage = TwoStageAgent(
        selector=agent,
        aimer=aimer,
        fire_threshold=env.fire_threshold,
        sample_every=sample_every,
        commit_steps=commit_steps,
        deterministic_selector=True,
        deterministic_aimer=aim_det,
    )

    renderer = PygameRenderer(title="Asteroid Defense - Trained Agent")
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
        base_action, _, _, _ = two_stage.step(obs, env)
        obs, reward, done, _, _ = env.step(base_action)
        total_reward += reward

        renderer.draw(env, reward=reward, total_reward=total_reward)

        if done:
            if seed_list:
                obs, _ = env.reset(seed=seed_list[seed_idx % len(seed_list)])
                seed_idx += 1
            else:
                obs, _ = env.reset()
            total_reward = 0.0
            two_stage.reset()

    renderer.close()


if __name__ == "__main__":
    run_agent(cfg_path='./configs/config_eval.yaml')
