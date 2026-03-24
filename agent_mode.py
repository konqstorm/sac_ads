import os
import yaml
import numpy as np
import torch

from core.env import AsteroidDefenseEnv
from core.models import Actor
from core.visual_pygame import PygameRenderer


def _load_env(cfg_path=os.path.join("configs", "config.yaml")):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
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


def run_agent(cfg_path=os.path.join("configs", "config.yaml"), weights_dir="weights", stochastic=True):
    env = _load_env(cfg_path)
    obs, _ = env.reset()
    actor = _load_actor(env, weights_dir=weights_dir)

    renderer = PygameRenderer(title="Asteroid Defense - Trained Agent")
    total_reward = 0.0

    running = True
    while running:
        running = renderer.process_events()
        action = _act_stochastic(actor, obs) if stochastic else _act_deterministic(actor, obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        renderer.draw(env, reward=reward, total_reward=total_reward)

        if done:
            obs, _ = env.reset()
            total_reward = 0.0

    renderer.close()


if __name__ == "__main__":
    run_agent(stochastic=True)
