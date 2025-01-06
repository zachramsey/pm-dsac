'''
Derived from: github.com/Jingliang-Duan/DSAC-v2/blob/main/networks/mlp.py
'''

import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, cfg):
        super(Actor, self).__init__()
        self.min_log_std = cfg["min_log_std"]
        self.max_log_std = cfg["max_log_std"]

        self.input = nn.Sequential(nn.Linear(cfg["latent_dim"], 256), nn.GELU())
        self.hidden1 = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.hidden2 = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.hidden3 = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.output1 = nn.Sequential(nn.Linear(256, 2), nn.GELU())

    def forward(self, s):
        s = s.reshape(s.size(0), -1)
        x = self.input(s)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.output1(x)
        mu, log_std = torch.chunk(x, chunks=2, dim=-1)
        std = torch.exp(torch.clamp(log_std, self.min_log_std, self.max_log_std))
        return mu, std
    