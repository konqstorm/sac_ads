import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import time

from core.env import AsteroidDefenseEnv
from core.vec_env import VecEnv
from core.models import Actor, Critic
from core.sac import SAC


# ---------------------------------------------------------------------------
# Policy helpers (probabilities -> baseline control)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save_training_plots(rewards, hulls, kills, out_dir="results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    x = np.arange(1, len(rewards) + 1)

    for values, label, fname in [
        (rewards, "reward",      "reward.png"),
        (hulls,   "hull_damage", "hull_damage.png"),
        (kills,   "kills",       "kills.png"),
    ]:
        plt.figure(figsize=(8, 4))
        plt.plot(x, values, label=label)
        # скользящее среднее для читаемости
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


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_agent(agent: SAC, out_dir="results/weights"):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(agent.actor.state_dict(),   os.path.join(out_dir, "actor.pt"))
    torch.save(agent.critic1.state_dict(), os.path.join(out_dir, "critic1.pt"))
    torch.save(agent.critic2.state_dict(), os.path.join(out_dir, "critic2.pt"))
    print(f"[save] weights → {out_dir}/")


# ---------------------------------------------------------------------------
# Critic evaluation
# ---------------------------------------------------------------------------

def evaluate_critic(agent, env, n_episodes=5, gamma=0.99, sample_every=5):
    """
    Собирает эпизоды, считает MC-returns и сравнивает
    с предсказаниями критика. Возвращает explained variance.
    """
    all_states, all_actions, all_returns = [], [], []

    for _ in range(n_episodes):
        s, _ = env.reset()
        episode = []  # (state, action, reward)

        done = False
        steps_since_sample = 0
        current_target = None
        while not done:
            raw_action = agent.act_deterministic(s)
            probs = _softmax(raw_action)

            if steps_since_sample == 0 or steps_since_sample >= sample_every:
                ordered = _ordered_asteroids(env)
                current_target = _sample_target_index(probs, len(ordered))
                steps_since_sample = 0

            base_action, fired = _baseline_action(env, current_target)
            s2, r, done, _, _ = env.step(base_action)
            episode.append((s, raw_action, r))
            s = s2

            if fired:
                steps_since_sample = sample_every  # force resample next step
            steps_since_sample += 1

        # MC return с конца эпизода
        G = 0.0
        for state, action, reward in reversed(episode):
            G = reward + gamma * G
            all_states.append(state)
            all_actions.append(action)
            all_returns.append(G)

    states  = torch.tensor(np.array(all_states),  dtype=torch.float32)
    actions = torch.tensor(np.array(all_actions), dtype=torch.float32)
    returns = torch.tensor(all_returns,            dtype=torch.float32)

    with torch.no_grad():
        q1 = agent.critic1(states, actions).squeeze()
        q2 = agent.critic2(states, actions).squeeze()
        q_pred = (q1 + q2) / 2.0

    # Explained variance: 1 - Var(returns - q_pred) / Var(returns)
    residual = returns - q_pred
    ev = 1.0 - residual.var() / (returns.var() + 1e-8)

    # Дополнительно: средняя абсолютная ошибка
    mae = residual.abs().mean().item()

    return {
        "explained_variance": ev.item(),
        "mae": mae,
        "q_mean": q_pred.mean().item(),
        "return_mean": returns.mean().item(),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_agent():
    with open(os.path.join("configs", "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    env_cfg   = cfg["env"]
    agent_cfg = cfg["agent"]
    train_cfg = cfg["train"]

    # --- seeds ---
    seed = train_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- vectorized env ---
    n_envs = train_cfg.get("n_envs", 2)
    seeds  = [seed + i * 100 for i in range(n_envs)]
    vec_env = VecEnv(env_cfg, n_envs=n_envs, seeds=seeds)

    obs_list = vec_env.reset()

    # --- eval env (одиночная, для чистой оценки) ---
    eval_env = AsteroidDefenseEnv(env_cfg)

    # --- agent ---
    obs_dim = vec_env.observation_space.shape[0]
    act_dim = 5

    actor   = Actor(obs_dim, act_dim)
    critic1 = Critic(obs_dim, act_dim)
    critic2 = Critic(obs_dim, act_dim)
    agent   = SAC(actor, critic1, critic2, agent_cfg)

    start_steps   = agent_cfg.get("start_steps",      5000)
    update_every  = agent_cfg.get("update_every",        4)
    updates_per_step = agent_cfg.get("updates_per_step", 4)
    plot_every    = train_cfg.get("plot_every",          10)
    critic_eval_every = train_cfg.get("critic_eval_every", 0)
    total_eps     = train_cfg.get("episodes",           400)
    sample_every  = agent_cfg.get("target_sample_every", 5)

    # Метрики собираем по эпизодам через eval_env (одиночный, честный)
    rewards_log, hulls_log, kills_log = [], [], []

    total_steps   = 0           # глобальный счётчик шагов по всем envs
    t_start = time.perf_counter()

    print(f"Starting training: {total_eps} episodes, {n_envs} envs, "
          f"update_every={update_every}, updates_per_step={updates_per_step}")
    print(f"Obs dim={obs_dim}, Act dim={act_dim}")
    print("-" * 60)

    for ep in range(total_eps):

        # --- сбор опыта через vec_env ---
        # Эпизод завершается, когда каждая env хотя бы один раз завершилась (done=True).
        steps_this_ep = vec_env.envs[0].max_steps
        done_once = [False for _ in range(n_envs)]

        # per-env target sampling state
        steps_since_sample = [0 for _ in range(n_envs)]
        current_target = [None for _ in range(n_envs)]

        for _ in range(steps_this_ep):
            # выбираем действия
            if total_steps < start_steps:
                raw_actions = [np.random.uniform(-1.0, 1.0, size=(5,)) for _ in range(n_envs)]
            else:
                raw_actions = [agent.act(s) for s in obs_list]

            # convert raw action -> probs -> baseline control
            actions = []
            for i in range(n_envs):
                probs = _softmax(raw_actions[i])
                if steps_since_sample[i] == 0 or steps_since_sample[i] >= sample_every:
                    ordered = _ordered_asteroids(vec_env.envs[i])
                    current_target[i] = _sample_target_index(probs, len(ordered))
                    steps_since_sample[i] = 0

                base_action, fired = _baseline_action(vec_env.envs[i], current_target[i])
                actions.append(base_action)

                if fired:
                    steps_since_sample[i] = sample_every
                steps_since_sample[i] += 1

            obs2_list, reward_list, done_list, _ = vec_env.step(actions)

            # кладём в буфер
            for i in range(n_envs):
                agent.buffer.add(
                    obs_list[i], raw_actions[i],
                    reward_list[i],
                    obs2_list[i],
                    float(done_list[i])
                )

            obs_list = obs2_list
            total_steps += n_envs  # n_envs шагов за одну итерацию

            # обновляем сети
            if (total_steps >= start_steps
                    and total_steps % update_every == 0):
                for _ in range(updates_per_step):
                    agent.update()

            # проверяем завершение эпизода по всем средам
            for i in range(n_envs):
                if done_list[i]:
                    done_once[i] = True
            if all(done_once):
                break

        # --- eval: один честный эпизод без шума ---
        s, _ = eval_env.reset(seed=seed + ep)
        ep_reward, ep_done = 0.0, False
        steps_since_sample_eval = 0
        current_target_eval = None
        while not ep_done:
            raw_action = agent.act_deterministic(s)
            probs = _softmax(raw_action)
            if steps_since_sample_eval == 0 or steps_since_sample_eval >= sample_every:
                ordered = _ordered_asteroids(eval_env)
                current_target_eval = _sample_target_index(probs, len(ordered))
                steps_since_sample_eval = 0

            base_action, fired = _baseline_action(eval_env, current_target_eval)
            s, r, ep_done, _, _ = eval_env.step(base_action)
            ep_reward += r

            if fired:
                steps_since_sample_eval = sample_every
            steps_since_sample_eval += 1

        rewards_log.append(ep_reward)
        hulls_log.append(eval_env.hull_damage)
        kills_log.append(eval_env.kills)

        avg_ep_time = (time.perf_counter() - t_start) / (ep + 1)
        print(f"ep {ep:4d} | "
              f"reward {ep_reward:7.2f} | "
              f"hull {eval_env.hull_damage:3d} | "
              f"kills {eval_env.kills:3d} | "
              f"buf {len(agent.buffer.buf):6d} | "
              f"steps {total_steps:7d} | "
              f"avg_ep_time {avg_ep_time:.2f}s")

        if critic_eval_every and ep % critic_eval_every == 0 and ep > 0:
            critic_stats = evaluate_critic(
                agent, eval_env, n_episodes=5,
                gamma=agent_cfg.get("gamma", 0.99),
                sample_every=sample_every
            )
            print(f"  [critic] EV={critic_stats['explained_variance']:.3f} "
                  f"MAE={critic_stats['mae']:.3f} "
                  f"Q_mean={critic_stats['q_mean']:.2f} "
                  f"R_mean={critic_stats['return_mean']:.2f}")

        if ep % plot_every == 0:
            _save_training_plots(rewards_log, hulls_log, kills_log)

    # финальные графики и веса
    _save_training_plots(rewards_log, hulls_log, kills_log)
    return agent


if __name__ == "__main__":
    agent = train_agent()
    save_agent(agent)
