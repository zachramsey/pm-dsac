'''
Derived from: github.com/Jingliang-Duan/DSAC-v2/blob/main/networks/mlp.py
'''

import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()
        window = cfg["window_size"]
        feat_dim = cfg["feat_dim"]

        layers = cfg["value_mlp_layers"]
        activation = getattr(nn, cfg["value_activation"])

        self.critic = nn.Sequential()
        self.critic.add_module("input", nn.Linear(window * (feat_dim + 1), layers[0]))
        self.critic.add_module("input_activation", activation())

        for i in range(1, len(layers)):
            self.critic.add_module(f"hidden_{i}", nn.Linear(layers[i - 1], layers[i]))
            self.critic.add_module(f"hidden_activation_{i}", activation())

        self.critic.add_module("output", nn.Linear(layers[-1], 2))

    def forward(self, s, a):
        a = a.unsqueeze(-1).expand(-1, s.size(1), -1)
        sa = torch.cat([s, a], dim=-1).view(s.size(0), -1)

        logits = self.critic(sa)
        q, std = torch.chunk(logits, chunks=2, dim=-1)
        log_std = torch.nn.functional.softplus(std)

        return q, log_std
