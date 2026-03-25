import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.enc = Encoder(input_dim, hidden=256)
        self.logits = nn.Linear(256, action_dim)

    def forward(self, x):
        # Выдаем сырые логиты для категориального распределения
        return self.logits(self.enc(x))

class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            # Критик оценивает Q-value для *каждой* возможной цели сразу
            nn.Linear(256, action_dim) 
        )

    def forward(self, x):
        return self.q(x)