import yaml
import numpy as np
from env import AsteroidDefenseEnv
from models import Actor, Critic
from sac import SAC

def train_agent():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    env = AsteroidDefenseEnv(cfg["env"])

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = Actor(obs_dim, act_dim)
    critic1 = Critic(obs_dim, act_dim)
    critic2 = Critic(obs_dim, act_dim)

    agent = SAC(actor, critic1, critic2, cfg["agent"])

    total_steps = 0
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

        print(f"ep {ep} reward {total_reward:.2f}")

    return agent
