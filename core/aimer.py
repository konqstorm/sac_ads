import os
import torch
import numpy as np

from core.models_continuous import ActorContinuous


class FrozenAimer:
    def __init__(self, obs_dim, action_dim, weights_dir, device="cpu"):
        self.device = device
        self.actor = ActorContinuous(obs_dim, action_dim).to(self.device)
        path = os.path.join(weights_dir, "actor.pt")
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        self.actor.eval()
        for p in self.actor.parameters():
            p.requires_grad = False

    def act(self, obs, deterministic=False):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.actor(obs_t)
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                action = torch.tanh(dist.sample())
        return action.cpu().numpy()[0]
