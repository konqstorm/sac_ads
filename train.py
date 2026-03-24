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
# Curriculum
# ---------------------------------------------------------------------------

def get_current_stage(episode: int, stages: list[dict]) -> dict:
    """Возвращает первую стадию, чей until_ep > episode."""
    for stage in stages:
        if episode < stage["until_ep"]:
            return stage
    return stages[-1]


def apply_curriculum(vec_env: VecEnv, stage: dict):
    vec_env.apply_stage(stage)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save_training_plots(rewards, hulls, kills, out_dir="plots"):
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

def save_agent(agent: SAC, out_dir="weights"):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(agent.actor.state_dict(),   os.path.join(out_dir, "actor.pt"))
    torch.save(agent.critic1.state_dict(), os.path.join(out_dir, "critic1.pt"))
    torch.save(agent.critic2.state_dict(), os.path.join(out_dir, "critic2.pt"))
    print(f"[save] weights → {out_dir}/")


# ---------------------------------------------------------------------------
# Critic evaluation
# ---------------------------------------------------------------------------

def evaluate_critic(agent, env, n_episodes=5, gamma=0.99):
    """
    Собирает эпизоды, считает MC-returns и сравнивает
    с предсказаниями критика. Возвращает explained variance.
    """
    all_states, all_actions, all_returns = [], [], []

    for _ in range(n_episodes):
        s, _ = env.reset()
        episode = []  # (state, action, reward)

        done = False
        while not done:
            a = agent.act_deterministic(s)
            s2, r, done, _, _ = env.step(a)
            episode.append((s, a, r))
            s = s2

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
    curriculum_stages = cfg.get("curriculum", [])

    # --- seeds ---
    seed = train_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- vectorized env ---
    n_envs = train_cfg.get("n_envs", 2)
    seeds  = [seed + i * 100 for i in range(n_envs)]
    vec_env = VecEnv(env_cfg, n_envs=n_envs, seeds=seeds)

    # Применяем первую стадию curriculum сразу
    if curriculum_stages:
        apply_curriculum(vec_env, get_current_stage(0, curriculum_stages))

    obs_list = vec_env.reset()

    # --- eval env (одиночная, для чистой оценки) ---
    eval_env = AsteroidDefenseEnv(env_cfg)

    # --- agent ---
    obs_dim = vec_env.observation_space.shape[0]
    act_dim = vec_env.action_space.shape[0]

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

    # Метрики собираем по эпизодам через eval_env (одиночный, честный)
    rewards_log, hulls_log, kills_log = [], [], []

    total_steps   = 0           # глобальный счётчик шагов по всем envs
    prev_stage_id = None
    t_start = time.perf_counter()

    print(f"Starting training: {total_eps} episodes, {n_envs} envs, "
          f"update_every={update_every}, updates_per_step={updates_per_step}")
    print(f"Obs dim={obs_dim}, Act dim={act_dim}")
    print("-" * 60)

    for ep in range(total_eps):

        # --- curriculum: обновляем среды при смене стадии ---
        if curriculum_stages:
            stage = get_current_stage(ep, curriculum_stages)
            stage_id = stage["until_ep"]
            if stage_id != prev_stage_id:
                apply_curriculum(vec_env, stage)
                # eval_env тоже обновляем
                for key, val in stage.items():
                    if key != "until_ep" and hasattr(eval_env, key):
                        setattr(eval_env, key, val)
                prev_stage_id = stage_id
                print(f"  [curriculum] ep={ep} → stage until_ep={stage_id} | "
                      f"ang_vel={stage.get('max_ang_vel','?')} "
                      f"asteroids={stage.get('max_asteroids','?')}")

        # --- сбор опыта через vec_env ---
        # Каждый "episode" здесь = max_steps шагов в каждой из n_envs сред.
        # Сброс среды при done происходит внутри vec_env.step() автоматически.
        steps_this_ep = vec_env.envs[0].max_steps

        for _ in range(steps_this_ep):
            # выбираем действия
            if total_steps < start_steps:
                actions = [vec_env.action_space.sample() for _ in range(n_envs)]
            else:
                actions = [agent.act(s) for s in obs_list]

            obs2_list, reward_list, done_list, _ = vec_env.step(actions)

            # кладём в буфер
            for i in range(n_envs):
                agent.buffer.add(
                    obs_list[i], actions[i],
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

        # --- eval: один честный эпизод без шума ---
        s, _ = eval_env.reset(seed=seed + ep)
        ep_reward, ep_done = 0.0, False
        while not ep_done:
            a = agent.act_deterministic(s)
            s, r, ep_done, _, _ = eval_env.step(a)
            ep_reward += r

        rewards_log.append(ep_reward)
        hulls_log.append(eval_env.hull_damage)
        kills_log.append(eval_env.kills)

        ang_vel_now = vec_env.envs[0].max_ang_vel
        avg_ep_time = (time.perf_counter() - t_start) / (ep + 1)
        print(f"ep {ep:4d} | "
              f"reward {ep_reward:7.2f} | "
              f"hull {eval_env.hull_damage:3d} | "
              f"kills {eval_env.kills:3d} | "
              f"buf {len(agent.buffer.buf):6d} | "
              f"steps {total_steps:7d} | "
              f"ang_vel {ang_vel_now:.2f} | "
              f"avg_ep_time {avg_ep_time:.2f}s")

        if critic_eval_every and ep % critic_eval_every == 0 and ep > 0:
            critic_stats = evaluate_critic(agent, eval_env, n_episodes=5, gamma=agent_cfg.get("gamma", 0.99))
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
