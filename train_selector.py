import os
import yaml
import numpy as np
import torch
import random
import time
import matplotlib.pyplot as plt

from core.env import AsteroidDefenseEnv
from core.models_discrete import Actor, Critic
from core.sac_discrete import SAC


def _save_training_plots(rewards, hulls, kills, out_dir="results/plots_selector"):
    os.makedirs(out_dir, exist_ok=True)
    x = np.arange(1, len(rewards) + 1)

    for values, label, fname in [
        (rewards, "reward",      "reward.png"),
        (hulls,   "hull_damage", "hull_damage.png"),
        (kills,   "kills",       "kills.png"),
    ]:
        plt.figure(figsize=(8, 4))
        plt.plot(x, values, label=label, alpha=0.6)
        if len(values) >= 10:
            window = np.convolve(values, np.ones(10) / 10, mode="valid")
            plt.plot(np.arange(10, len(values) + 1), window,
                     label=f"{label} (avg10)", linewidth=2)
        plt.xlabel("episode")
        plt.ylabel(label)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()


def _select_best_slot(env):
    best_idx = None
    best_score = None
    for i, a in enumerate(env.asteroid_slots):
        if a is None:
            continue
        r = a["pos"].astype(np.float32)
        v = a["vel"].astype(np.float32)
        t = env._solve_intercept(r, v, env.projectile_speed)
        aim_point = r + v * t if t is not None else r
        target_yaw, target_pitch = env._direction_to_yaw_pitch(aim_point)
        err_yaw = target_yaw - env.yaw
        err_pitch = target_pitch - env.pitch
        score = abs(err_yaw) + abs(err_pitch)
        if best_idx is None or score < best_score:
            best_idx = i
            best_score = score
    return best_idx


def _baseline_action(env, target_idx, aim_eps=0.02):
    if target_idx is None:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32), False
    if target_idx < 0 or target_idx >= len(env.asteroid_slots):
        return np.array([0.0, 0.0, -1.0], dtype=np.float32), False
    target = env.asteroid_slots[target_idx]
    if target is None:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32), False

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

    fired = abs(err_yaw) < aim_eps and abs(err_pitch) < aim_eps
    fire_action = 1.0 if fired else -1.0

    return np.array([yaw_action, pitch_action, fire_action], dtype=np.float32), fired


def save_selector(agent: SAC, out_dir="results/weights_selector"):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(agent.actor.state_dict(),   os.path.join(out_dir, "actor.pt"))
    torch.save(agent.critic1.state_dict(), os.path.join(out_dir, "critic1.pt"))
    torch.save(agent.critic2.state_dict(), os.path.join(out_dir, "critic2.pt"))
    print(f"[save] selector weights → {out_dir}/")


def train_selector(cfg_path=os.path.join("configs", "config_select.yaml")):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]
    train_cfg = cfg["train"]

    seed = train_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = AsteroidDefenseEnv(env_cfg)

    obs_dim = env.observation_space.shape[0]
    act_dim = int(env_cfg.get("max_asteroids", 5))

    actor = Actor(obs_dim, act_dim)
    critic1 = Critic(obs_dim, act_dim)
    critic2 = Critic(obs_dim, act_dim)
    agent = SAC(actor, critic1, critic2, agent_cfg)

    start_steps = agent_cfg.get("start_steps", 3000)
    update_every = agent_cfg.get("update_every", 4)
    updates_per_step = agent_cfg.get("updates_per_step", 2)

    reward_correct = 5.0
    reward_ghost = -5.0
    penalty_switch = -0.5

    total_eps = train_cfg.get("episodes", 200)
    total_steps = 0
    t_start = time.perf_counter()

    rewards_log, hulls_log, kills_log = [], [], []
    plot_every = train_cfg.get("plot_every", 10)
    plots_dir = train_cfg.get("plots_dir_selector", "results/plots_selector")

    print(f"Starting selector training: {total_eps} episodes")
    print(f"Obs dim={obs_dim}, Act dim={act_dim}")
    print("-" * 60)

    for ep in range(total_eps):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        ep_env_reward = 0.0
        prev_action = None

        while not done:
            if total_steps < start_steps:
                action = np.random.randint(0, act_dim)
            else:
                action = agent.act(obs)

            best_idx = _select_best_slot(env)

            reward = 0.0
            valid = True
            if action < 0 or action >= act_dim:
                valid = False
            else:
                if env.asteroid_slots[action] is None:
                    valid = False

            if not valid:
                reward += reward_ghost
                target_idx = None
            else:
                target_idx = int(action)
                if best_idx is not None and target_idx == best_idx:
                    reward += reward_correct

            if prev_action is not None and action != prev_action:
                reward += penalty_switch
            prev_action = action

            env_action, _ = _baseline_action(env, target_idx)
            obs2, env_reward, done, _, _ = env.step(env_action)

            agent.buffer.add(obs, action, reward, obs2, float(done))

            if not valid:
                env_reward += reward_ghost
            if prev_action is not None and action != prev_action:
                env_reward += penalty_switch

            obs = obs2
            ep_reward += reward
            ep_env_reward += env_reward
            total_steps += 1

            if total_steps >= start_steps and total_steps % update_every == 0:
                for _ in range(updates_per_step):
                    agent.update()

        avg_ep_time = (time.perf_counter() - t_start) / (ep + 1)
        print(f"ep {ep:4d} | reward {ep_reward:7.2f} | "
              f"hull {env.hull_damage:3d} | kills {env.kills:3d} | "
              f"buf {len(agent.buffer.buf):6d} | steps {total_steps:7d} | "
              f"avg_ep_time {avg_ep_time:.2f}s")

        rewards_log.append(ep_env_reward)
        hulls_log.append(env.hull_damage)
        kills_log.append(env.kills)

        if ep % plot_every == 0:
            _save_training_plots(rewards_log, hulls_log, kills_log, out_dir=plots_dir)

    _save_training_plots(rewards_log, hulls_log, kills_log, out_dir=plots_dir)
    save_dir = train_cfg.get("save_dir_selector", "results/weights_selector")
    save_selector(agent, out_dir=save_dir)
    return agent


if __name__ == "__main__":
    train_selector()
