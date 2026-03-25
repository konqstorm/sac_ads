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

class SAC:
    def __init__(self, actor, critic1, critic2, config):
        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.target1 = copy.deepcopy(critic1)
        self.target2 = copy.deepcopy(critic2)

        self.gamma = config["gamma"]
        self.tau = config["tau"]

        # Эвристика энтропии для дискретного пространства
        act_dim = actor.logits.out_features
        self.target_entropy = -np.log(1.0 / act_dim) * 0.98
        
        init_alpha = config.get("alpha", 0.2)
        self.log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True)

        lr_actor = config.get("lr_actor", 1e-4)
        lr_critic = config.get("lr_critic", 3e-4)

        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr_actor)
        self.opt_actor = torch.optim.Adam(actor.parameters(), lr=lr_actor)
        self.opt_critic1 = torch.optim.Adam(critic1.parameters(), lr=lr_critic)
        self.opt_critic2 = torch.optim.Adam(critic2.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer(config["buffer_size"])
        self.batch_size = config["batch_size"]

    def act(self, state):
        state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(state)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action.item()

    def act_deterministic(self, state):
        state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(state)
            action = torch.argmax(logits, dim=-1)
        return action.item()

    def update(self):
        if len(self.buffer.buf) < self.batch_size: return

        s, a, r, s2, d = self.buffer.sample(self.batch_size)

        s = torch.tensor(s, dtype=torch.float32)
        s2 = torch.tensor(s2, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.long).unsqueeze(1) # Дискретное действие!
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            logits2 = self.actor(s2)
            probs2 = F.softmax(logits2, dim=-1)
            log_probs2 = F.log_softmax(logits2, dim=-1)
            
            q1_t = self.target1(s2)
            q2_t = self.target2(s2)
            q_t = torch.min(q1_t, q2_t)
            
            # Ожидаемое V-value (взвешенная сумма по всем действиям)
            v_t = (probs2 * (q_t - alpha * log_probs2)).sum(dim=-1, keepdim=True)
            target = r + self.gamma * (1 - d) * v_t

        # Получаем Q-значения только для выбранных агентом действий
        q1 = self.critic1(s).gather(1, a)
        q2 = self.critic2(s).gather(1, a)

        loss_q1 = F.mse_loss(q1, target)
        loss_q2 = F.mse_loss(q2, target)

        self.opt_critic1.zero_grad()
        loss_q1.backward()
        self.opt_critic1.step()

        self.opt_critic2.zero_grad()
        loss_q2.backward()
        self.opt_critic2.step()

        # Обновление актора
        logits = self.actor(s)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            q1_pi = self.critic1(s)
            q2_pi = self.critic2(s)
            q_pi = torch.min(q1_pi, q2_pi)

        inside_term = alpha * log_probs - q_pi
        loss_actor = (probs * inside_term).sum(dim=-1).mean()

        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

        # Обновление альфы
        alpha_loss = -(self.log_alpha * (probs.detach() * (log_probs + self.target_entropy).detach()).sum(dim=-1)).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        # Soft update
        for p, tp in zip(self.critic1.parameters(), self.target1.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic2.parameters(), self.target2.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)