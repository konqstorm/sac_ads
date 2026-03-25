import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ActorContinuous(nn.Module):
    def __init__(self, input_dim, action_dim, hidden=256):
        super().__init__()
        self.enc = Encoder(input_dim, hidden=hidden)
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, x):
        h = self.enc(x)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -5, 2)
        return mean, log_std


class CriticContinuous(nn.Module):
    def __init__(self, input_dim, action_dim, hidden=256):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x, a):
        return self.q(torch.cat([x, a], dim=-1))
