import os
import yaml
import numpy as np
import torch
import random
import time
import matplotlib.pyplot as plt

from core.env import AsteroidDefenseEnv
from core.models_continuous import ActorContinuous, CriticContinuous
from core.sac_continuous import SACContinuous
from core.aim_utils import extract_aim_obs


def _save_training_plots(rewards, hulls, kills, out_dir="results/plots_aim"):
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


def save_aimer(agent: SACContinuous, out_dir="results/weights_aim"):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(agent.actor.state_dict(), os.path.join(out_dir, "actor.pt"))
    torch.save(agent.critic1.state_dict(), os.path.join(out_dir, "critic1.pt"))
    torch.save(agent.critic2.state_dict(), os.path.join(out_dir, "critic2.pt"))
    print(f"[save] aim weights → {out_dir}/")


def train_aimer(cfg_path=os.path.join("configs", "config_aim.yaml")):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]
    train_cfg = cfg["train"]
    aim_cfg = cfg.get("aim", {})

    seed = train_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = AsteroidDefenseEnv(env_cfg)

    obs_dim = aim_cfg.get("obs_dim", 7)
    act_dim = aim_cfg.get("action_dim", 3)

    actor = ActorContinuous(obs_dim, act_dim)
    critic1 = CriticContinuous(obs_dim, act_dim)
    critic2 = CriticContinuous(obs_dim, act_dim)
    agent = SACContinuous(actor, critic1, critic2, agent_cfg)

    # optional resume
    resume = bool(train_cfg.get("resume", False))
    resume_dir = train_cfg.get("resume_dir", train_cfg.get("save_dir", "results/weights_aim"))
    if resume:
        actor_path = os.path.join(resume_dir, "actor.pt")
        critic1_path = os.path.join(resume_dir, "critic1.pt")
        critic2_path = os.path.join(resume_dir, "critic2.pt")
        if os.path.exists(actor_path):
            actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
        if os.path.exists(critic1_path):
            critic1.load_state_dict(torch.load(critic1_path, map_location="cpu"))
        if os.path.exists(critic2_path):
            critic2.load_state_dict(torch.load(critic2_path, map_location="cpu"))
        print(f"[resume] loaded weights from {resume_dir}")

    start_steps = agent_cfg.get("start_steps", 1000)
    update_every = agent_cfg.get("update_every", 4)
    updates_per_step = agent_cfg.get("updates_per_step", 2)

    dense_weight = aim_cfg.get("dense_weight", 0.01)
    dense_scale = aim_cfg.get("dense_scale", 0.1)

    total_eps = train_cfg.get("episodes", 200)
    total_steps = 0
    t_start = time.perf_counter()
    rewards_log, hulls_log, kills_log = [], [], []
    plot_every = train_cfg.get("plot_every", 10)
    plots_dir = train_cfg.get("plots_dir", "results/plots_aim")

    print(f"Starting aim training: {total_eps} episodes")
    print(f"Aim obs dim={obs_dim}, act dim={act_dim}")
    print("-" * 60)

    for ep in range(total_eps):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        ep_reward_old = 0.0

        while not done:
            aim_obs = extract_aim_obs(obs, 0)

            if total_steps < start_steps:
                action = np.random.uniform(-1.0, 1.0, size=(act_dim,)).astype(np.float32)
            else:
                action = agent.act(aim_obs)

            obs2, reward, done, _, _ = env.step(action)
            aim_obs2 = extract_aim_obs(obs2, 0) # Берем наблюдение ПОСЛЕ шага!

            # old reward: alignment between gun dir and analytic aim dir
            reward_old = 0.0
            target = env.asteroid_slots[0] if env.asteroid_slots else None
            if target is not None:
                r = target["pos"].astype(np.float32)
                v = target["vel"].astype(np.float32)
                t = env._solve_intercept(r, v, env.projectile_speed)
                aim_point = r + v * t if t is not None else r
                ideal_dir = aim_point / (np.linalg.norm(aim_point) + 1e-8)
                gun_dir = env._gun_direction().astype(np.float32)
                reward_old = -float(np.linalg.norm(gun_dir - ideal_dir))
            if action[2] > env.fire_threshold:
                reward_old += 0.01

            # new reward: distance to baseline action
            reward = 0.0
            if target is not None:
                target_yaw, target_pitch = env._direction_to_yaw_pitch(aim_point)
                half_fov = env.fov / 2.0
                target_yaw = np.clip(target_yaw, -half_fov, half_fov)
                target_pitch = np.clip(target_pitch, -half_fov, half_fov)

                err_yaw = target_yaw - env.yaw
                err_pitch = target_pitch - env.pitch
                step = env.max_ang_vel * env.dt
                yaw_action = np.clip(err_yaw / step, -1.0, 1.0)
                pitch_action = np.clip(err_pitch / step, -1.0, 1.0)
                fire_action = 1.0 #if (abs(err_yaw) < 0.02 and abs(err_pitch) < 0.02) else -1.0
                baseline_action = np.array([yaw_action, pitch_action, fire_action], dtype=np.float32)
                reward = -float(np.linalg.norm(action - baseline_action))

            agent.buffer.add(aim_obs, action, reward, aim_obs2, float(done))

            obs = obs2
            ep_reward += reward
            ep_reward_old += reward_old
            total_steps += 1

            if total_steps >= start_steps and total_steps % update_every == 0:
                for _ in range(updates_per_step):
                    agent.update()

        avg_ep_time = (time.perf_counter() - t_start) / (ep + 1)
        print(f"ep {ep:4d} | reward_new {ep_reward:7.2f} | "
              f"hull {env.hull_damage:3d} | kills {env.kills:3d} | "
              f"buf {len(agent.buffer.buf):6d} | steps {total_steps:7d} | "
              f"avg_ep_time {avg_ep_time:.2f}s")
        rewards_log.append(ep_reward_old)
        hulls_log.append(env.hull_damage)
        kills_log.append(env.kills)

        if ep % plot_every == 0:
            _save_training_plots(rewards_log, hulls_log, kills_log, out_dir=plots_dir)

    save_dir = train_cfg.get("save_dir", "results/weights_aim")
    save_aimer(agent, out_dir=save_dir)
    _save_training_plots(rewards_log, hulls_log, kills_log, out_dir=plots_dir)
    return agent


if __name__ == "__main__":
    train_aimer()
