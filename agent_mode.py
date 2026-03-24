import os
import yaml
import numpy as np
import torch

from core.env import AsteroidDefenseEnv
from core.models import Actor
from core.visual_pygame import PygameRenderer


def _softmax(x):
    x = np.array(x, dtype=np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    if s <= 1e-8:
        return np.ones_like(x) / len(x)
    return e / s


def _ordered_asteroids(env):
    if not env.asteroids:
        return []
    times = []
    for a in env.asteroids:
        t = env._solve_intercept(a["pos"], a["vel"], env.projectile_speed)
        times.append(t if t is not None else 1e6)
    order = np.argsort(times)
    return [env.asteroids[i] for i in order]


def _baseline_action(env, target_idx, aim_eps=0.02):
    ordered = _ordered_asteroids(env)
    if not ordered:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32), False

    if target_idx is None:
        target_idx = 0
    target_idx = int(np.clip(target_idx, 0, len(ordered) - 1))
    target = ordered[target_idx]

    r = target["pos"].astype(np.float32)
    v = target["vel"].astype(np.float32)
    t = env._solve_intercept(r, v, env.projectile_speed)
    aim_point = r + v * t if t is not None else r

    target_yaw, target_pitch = env._direction_to_yaw_pitch(aim_point)
    half_fov = env.fov / 2.0
    target_yaw = np.clip(target_yaw, -half_fov, half_fov)
    target_pitch = np.clip(target_pitch, -half_fov, half_fov)

    err_yaw = target_yaw - env.yaw
    err_pitch = target_pitch - env.pitch

    step = env.max_ang_vel * env.dt
    yaw_action = np.clip(err_yaw / step, -1.0, 1.0)
    pitch_action = np.clip(err_pitch / step, -1.0, 1.0)

    fire_action = -1.0
    fired = False
    if abs(err_yaw) < aim_eps and abs(err_pitch) < aim_eps:
        fire_action = 1.0
        fired = True

    return np.array([yaw_action, pitch_action, fire_action], dtype=np.float32), fired


def _sample_target_index(probs, n_available):
    if n_available <= 0:
        return None
    probs = np.array(probs[:n_available], dtype=np.float32)
    if probs.sum() <= 1e-6:
        probs = np.ones(n_available, dtype=np.float32) / n_available
    else:
        probs = probs / probs.sum()
    return int(np.random.choice(np.arange(n_available), p=probs))


def _load_env(cfg_path=os.path.join("configs", "config.yaml")):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return AsteroidDefenseEnv(cfg["env"])


def _load_actor(env, weights_dir="weights"):
    obs_dim = env.observation_space.shape[0]
    act_dim = 5
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


def run_agent(cfg_path=os.path.join("configs", "config.yaml"), weights_dir="results/weights", stochastic=True):
    env = _load_env(cfg_path)
    obs, _ = env.reset()
    actor = _load_actor(env, weights_dir=weights_dir)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    sample_every = cfg.get("agent", {}).get("target_sample_every", 5)

    renderer = PygameRenderer(title="Asteroid Defense - Trained Agent")
    total_reward = 0.0

    running = True
    steps_since_sample = 0
    current_target = None
    while running:
        running = renderer.process_events()
        raw_action = _act_stochastic(actor, obs) if stochastic else _act_deterministic(actor, obs)
        probs = _softmax(raw_action)

        if steps_since_sample == 0 or steps_since_sample >= sample_every:
            ordered = _ordered_asteroids(env)
            current_target = _sample_target_index(probs, len(ordered))
            steps_since_sample = 0

        base_action, fired = _baseline_action(env, current_target)
        obs, reward, done, _, _ = env.step(base_action)
        total_reward += reward

        renderer.draw(env, reward=reward, total_reward=total_reward)

        if done:
            obs, _ = env.reset()
            total_reward = 0.0
            steps_since_sample = 0
            current_target = None
        if fired:
            steps_since_sample = sample_every
        steps_since_sample += 1

    renderer.close()


if __name__ == "__main__":
    run_agent(stochastic=True)
