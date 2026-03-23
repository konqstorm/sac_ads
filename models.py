import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.enc = Encoder(input_dim)

        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

    def forward(self, x):
        h = self.enc(x)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -5, 2)
        return mean, log_std


class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(input_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, a):
        return self.q(torch.cat([x, a], dim=-1))