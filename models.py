import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        # Делаем сеть шире и глубже для аппроксимации физики
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.enc = Encoder(input_dim, hidden=256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, x):
        h = self.enc(x)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -5, 2)
        return mean, log_std

class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(input_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, a):
        return self.q(torch.cat([x, a], dim=-1))