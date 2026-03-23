import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from env import AsteroidDefenseEnv
from models import Actor, Critic
from sac import SAC

def train_agent():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    env = AsteroidDefenseEnv(cfg["env"])
    eval_env = AsteroidDefenseEnv(cfg["env"])

    seed = cfg.get("train", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = Actor(obs_dim, act_dim)
    critic1 = Critic(obs_dim, act_dim)
    critic2 = Critic(obs_dim, act_dim)

    agent = SAC(actor, critic1, critic2, cfg["agent"])

    total_steps = 0
    rewards = []
    hulls = []
    kills = []
    eval_rewards = []
    eval_hulls = []
    eval_kills = []

    for ep in range(cfg["train"]["episodes"]):
        s, _ = env.reset()
        total_reward = 0
        for _ in range(env.max_steps):
            if total_steps < cfg["agent"]["start_steps"]:
                a = env.action_space.sample() # Случайное действие
            else:
                a = agent.act(s)              # Действие агента
                
            s2, r, d, _, _ = env.step(a)

            agent.buffer.add(s, a, r, s2, d)
            
            # Обновляем сети ТОЛЬКО после фазы стартового сбора данных
            if total_steps >= cfg["agent"]["start_steps"]:
                agent.update()

            s = s2
            total_reward += r
            total_steps += 1

            if d:
                break

        rewards.append(total_reward)
        hulls.append(env.hull_damage)
        kills.append(env.kills)

        msg = f"ep {ep} reward {total_reward:.2f} hull_damage {env.hull_damage} kills {env.kills}"
        print(msg)

        _save_training_plots(rewards, hulls, kills, eval_rewards, eval_hulls, eval_kills, out_dir="plots")

    return agent


def save_agent(agent, out_dir="weights"):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(agent.actor.state_dict(), os.path.join(out_dir, "actor.pt"))
    torch.save(agent.critic1.state_dict(), os.path.join(out_dir, "critic1.pt"))
    torch.save(agent.critic2.state_dict(), os.path.join(out_dir, "critic2.pt"))


def _save_training_plots(rewards, hulls, kills, eval_rewards, eval_hulls, eval_kills, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    x = np.arange(1, len(rewards) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(x, rewards, label="reward")
    if eval_rewards:
        plt.plot(np.arange(1, len(eval_rewards) + 1), eval_rewards, label="eval_reward")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(x, hulls, label="hull_damage")
    if eval_hulls:
        plt.plot(np.arange(1, len(eval_hulls) + 1), eval_hulls, label="eval_hull_damage")
    plt.xlabel("episode")
    plt.ylabel("hull_damage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hull_damage.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(x, kills, label="kills")
    if eval_kills:
        plt.plot(np.arange(1, len(eval_kills) + 1), eval_kills, label="eval_kills")
    plt.xlabel("episode")
    plt.ylabel("kills")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kills.png"))
    plt.close()


if __name__ == "__main__":
    agent = train_agent()
    save_agent(agent)
