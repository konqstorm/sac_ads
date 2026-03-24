import os
import numpy as np
import torch

from env import AsteroidDefenseEnv
from gif_recorder import GIFRecorder
from models import Actor
from runtime_options import (
    load_config,
    load_ursina_loop,
    resolve_do_gif,
    resolve_fps,
    resolve_gif_directory,
    resolve_gif_fps,
    resolve_gif_name,
    resolve_renderer,
    resolve_stochastic_agent,
)
from visual_pygame import PygameRenderer


def _load_env(cfg):
    return AsteroidDefenseEnv(cfg["env"])


def _load_actor(env, weights_dir="weights"):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    actor = Actor(obs_dim, act_dim)
    path = os.path.join(weights_dir, "actor.pt")
    actor.load_state_dict(torch.load(path, map_location="cpu"))
    actor.eval()
    return actor


def _act_deterministic(actor, obs):
    with torch.no_grad():
        state = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
        mean, _ = actor(state)
        action = torch.tanh(mean)
    return action.numpy()[0]


def _act_stochastic(actor, obs):
    with torch.no_grad():
        state = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
        mean, log_std = actor(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = torch.tanh(dist.rsample())
    return action.numpy()[0]


def run_agent(cfg_path="config.yaml", weights_dir="weights", stochastic=True, renderer=None):
    cfg = load_config(cfg_path)
    env = _load_env(cfg)
    obs, _ = env.reset()
    actor = _load_actor(env, weights_dir=weights_dir)
    renderer_name = resolve_renderer(cfg, renderer=renderer)
    fps = resolve_fps(cfg, default=30)
    stochastic = resolve_stochastic_agent(cfg, stochastic=stochastic, default=True)
    gif_fps = resolve_gif_fps(cfg, default=fps)
    gif_recorder = GIFRecorder(
        enabled=resolve_do_gif(cfg, default=False),
        directory=resolve_gif_directory(cfg, default="tmp_gif"),
        name=resolve_gif_name(cfg, default="agent_run.gif"),
        fps=gif_fps,
    )

    if renderer_name == "3d":
        run_ursina_loop = load_ursina_loop()

        def _act(current_obs):
            if stochastic:
                return _act_stochastic(actor, current_obs)
            return _act_deterministic(actor, current_obs)

        run_ursina_loop(
            env=env,
            title="Asteroid Defense - Trained Agent (3D)",
            fps=fps,
            action_fn=_act,
            initial_obs=obs,
            extra_lines_fn=lambda _obs: [
                f"Policy: {'stochastic' if stochastic else 'deterministic'}"
            ],
            gif_recorder=gif_recorder,
        )
        return

    renderer = PygameRenderer(title="Asteroid Defense - Trained Agent", gif_recorder=gif_recorder)
    total_reward = 0.0

    running = True
    while running:
        running = renderer.process_events()
        action = _act_stochastic(actor, obs) if stochastic else _act_deterministic(actor, obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        renderer.draw(env, reward=reward, total_reward=total_reward, fps=fps)

        if done:
            obs, _ = env.reset()
            total_reward = 0.0

    renderer.close()


if __name__ == "__main__":
    run_agent(stochastic=None, renderer=None)
