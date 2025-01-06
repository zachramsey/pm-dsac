'''
Derived from: github.com/Jingliang-Duan/DSAC-v2/blob/main/networks/mlp.py
'''

import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()
        
        self.input = nn.Sequential(nn.Linear(cfg["latent_dim"]+1, 256), nn.GELU())
        self.hidden1 = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.hidden2 = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.hidden3 = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.output1 = nn.Sequential(nn.Linear(256, 2), nn.GELU())

    def forward(self, s, a):
        s = s.reshape(s.size(0), -1)
        x = torch.cat([s, a], dim=-1)
        x = self.input(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.output1(x)
        mu, std = torch.chunk(x, chunks=2, dim=-1)
        log_std = nn.functional.softplus(std)
        return mu, log_std
