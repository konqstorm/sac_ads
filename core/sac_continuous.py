import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import copy

class ReplayBuffer:
    def __init__(self, size):
        self.buf = deque(maxlen=size)

    def add(self, *args):
        self.buf.append(args)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return map(np.array, zip(*batch))

class SACContinuous:
    def __init__(self, actor, critic1, critic2, config):
        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.target1 = copy.deepcopy(critic1)
        self.target2 = copy.deepcopy(critic2)

        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)

        lr_actor = config.get("lr_actor", 3e-4)
        lr_critic = config.get("lr_critic", 3e-4)

        self.opt_actor = torch.optim.Adam(actor.parameters(), lr=lr_actor)
        self.opt_critic1 = torch.optim.Adam(critic1.parameters(), lr=lr_critic)
        self.opt_critic2 = torch.optim.Adam(critic2.parameters(), lr=lr_critic)

        # Автоматическая настройка энтропии (Auto-Alpha)
        action_dim = actor.mean.out_features
        self.target_entropy = -action_dim  # Эвристика: -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr_actor)

        self.buffer = ReplayBuffer(config.get("buffer_size", 100000))
        self.batch_size = config.get("batch_size", 256)

    def act(self, state):
        state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.actor(state)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = torch.tanh(dist.rsample())
        return action.numpy()[0]

    def act_deterministic(self, state):
        state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, _ = self.actor(state)
            action = torch.tanh(mean)
        return action.numpy()[0]

    def update(self):
        if len(self.buffer.buf) < self.batch_size:
            return

        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s = torch.tensor(s, dtype=torch.float32)
        s2 = torch.tensor(s2, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.float32)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            m, ls = self.actor(s2)
            std = ls.exp()
            dist = torch.distributions.Normal(m, std)
            u = dist.rsample()
            a2 = torch.tanh(u)
            logp2 = dist.log_prob(u).sum(-1, keepdim=True)
            logp2 -= torch.log(1 - a2.pow(2) + 1e-6).sum(-1, keepdim=True)

            q1_t = self.target1(s2, a2)
            q2_t = self.target2(s2, a2)
            q_t = torch.min(q1_t, q2_t) - alpha * logp2

            target = r + self.gamma * (1 - d) * q_t

        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)

        loss_q1 = F.mse_loss(q1, target)
        loss_q2 = F.mse_loss(q2, target)

        self.opt_critic1.zero_grad()
        loss_q1.backward()
        self.opt_critic1.step()

        self.opt_critic2.zero_grad()
        loss_q2.backward()
        self.opt_critic2.step()

        # Обучение Актора
        m, ls = self.actor(s)
        std = ls.exp()
        dist = torch.distributions.Normal(m, std)
        u_new = dist.rsample()
        a_new = torch.tanh(u_new)
        logp = dist.log_prob(u_new).sum(-1, keepdim=True)
        logp -= torch.log(1 - a_new.pow(2) + 1e-6).sum(-1, keepdim=True)

        q1_new = self.critic1(s, a_new)
        q2_new = self.critic2(s, a_new)

        loss_actor = (alpha * logp - torch.min(q1_new, q2_new)).mean()

        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

        # Обучение Alpha (Настройка энтропии)
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        # soft update
        for p, tp in zip(self.critic1.parameters(), self.target1.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic2.parameters(), self.target2.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)