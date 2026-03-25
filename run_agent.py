import os
import torch

from core.env import AsteroidDefenseEnv
from visuals.gif_recorder import GIFRecorder
from core.models_discrete import Actor, Critic
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
from core.sac_discrete import SAC
from core.aimer import FrozenAimer
from core.two_stage_agent import TwoStageAgent
from visuals.visual_pygame import PygameRenderer


def _load_env(cfg):
    return AsteroidDefenseEnv(cfg["env"])


def _seed_list(cfg):
    visual_cfg = cfg.get("visual", {})
    seeds = cfg.get("visual_seeds", visual_cfg.get("seeds"))
    return list(seeds) if seeds else []


def _load_agent(env, cfg, weights_dir=None):
    obs_dim = env.observation_space.shape[0]
    act_dim = int(cfg.get("env", {}).get("max_asteroids", 5))
    actor = Actor(obs_dim, act_dim)
    critic1 = Critic(obs_dim, act_dim)
    critic2 = Critic(obs_dim, act_dim)
    agent = SAC(actor, critic1, critic2, cfg.get("agent", {}))
    if weights_dir is None:
        weights_dir = cfg.get("train", {}).get("save_dir_selector", "results/weights_selector")
    path = os.path.join(weights_dir, "actor.pt")
    actor.load_state_dict(torch.load(path, map_location="cpu"))
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


def run_agent(cfg_path="configs/config_eval.yaml", renderer=None, weights_dir=None):
    cfg = load_config(cfg_path)
    env = _load_env(cfg)
    renderer_name = resolve_renderer(cfg, renderer=renderer)
    fps = resolve_fps(cfg, default=30)
    gif_fps = resolve_gif_fps(cfg, default=fps)
    gif_recorder = GIFRecorder(
        enabled=resolve_do_gif(cfg, default=False),
        directory=resolve_gif_directory(cfg, default="tmp_gif"),
        name=resolve_gif_name(cfg, default="agent_run.gif"),
        fps=gif_fps,
    )

    agent = _load_agent(env, cfg, weights_dir=weights_dir)
    sample_every = cfg.get("agent", {}).get("target_sample_every", 5)
    commit_steps = cfg.get("agent", {}).get("commit_steps", 15)

    aimer_cfg = cfg.get("aimer", {})
    aim_obs_dim = aimer_cfg.get("obs_dim", 7)
    aim_act_dim = aimer_cfg.get("action_dim", 3)
    aim_weights_dir = aimer_cfg.get("weights_dir", "results/weights_aim")
    aim_det = bool(aimer_cfg.get("deterministic_eval", True))
    aimer = FrozenAimer(aim_obs_dim, aim_act_dim, aim_weights_dir)

    two_stage = TwoStageAgent(
        selector=agent,
        aimer=aimer,
        fire_threshold=env.fire_threshold,
        sample_every=sample_every,
        commit_steps=commit_steps,
        deterministic_selector=True,
        deterministic_aimer=aim_det,
    )

    seeds = _seed_list(cfg)
    seed_idx = 0

    def reset_env():
        nonlocal seed_idx
        two_stage.reset()
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
            title="Asteroid Defense - Agent (3D)",
            fps=fps,
            initial_obs=obs,
            on_episode_reset=reset_env,
            action_fn=lambda current_obs: two_stage.step(current_obs, env)[0],
            gif_recorder=gif_recorder,
        )
        return

    renderer = PygameRenderer(title="Asteroid Defense - Trained Agent", gif_recorder=gif_recorder)
    total_reward = 0.0

    running = True
    while running:
        running = renderer.process_events()
        base_action, _, _, _ = two_stage.step(obs, env)
        obs, reward, done, _, _ = env.step(base_action)
        total_reward += reward

        renderer.draw(env, reward=reward, total_reward=total_reward, fps=fps)

        if done:
            obs = reset_env()
            total_reward = 0.0

    renderer.close()


if __name__ == "__main__":
    run_agent()
