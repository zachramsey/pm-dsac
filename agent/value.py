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
        input_dim = window * (feat_dim + 1)

        self.input = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU())
        self.hidden1 = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.hidden2 = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.hidden3 = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.output1 = nn.Sequential(nn.Linear(256, 2), nn.GELU())
        self.output2 = nn.Linear(cfg["asset_dim"], 1)

    def forward(self, s, a):
        a = a.unsqueeze(-1).expand(-1, s.size(1), -1)
        x = torch.cat([s, a], dim=-1).view(s.size(0), -1)

        # print(x)
        x = self.input(x)
        # print(x)
        x = self.hidden1(x)
        # print(x)
        x = self.hidden2(x)
        # print(x)
        x = self.hidden3(x)
        # print(x)
        x = self.output1(x)
        # print(x)
        x = x.transpose(0, 1)
        # print(x)
        x = self.output2(x)
        # print(x)
        x = x.transpose(0, 1)
        # print(x)
        q, std = torch.chunk(x, chunks=2, dim=-1)
        # print(q, std)

        return q, std
