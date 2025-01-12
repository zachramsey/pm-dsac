'''
Derived from: github.com/Jingliang-Duan/DSAC-v2/blob/main/utils/act_distribution_cls.py
'''

import torch
import torch.distributions as D

EPS = 1e-6

class TanhGaussDistribution:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

        self.gauss_distribution = D.Independent(
            base_distribution=D.Normal(self.mean, self.std),
            reinterpreted_batch_ndims=1,
        )
        self.act_max = torch.tensor([1.0])
        self.act_min = torch.tensor([-1.0])

    def sample(self):
        action = self.gauss_distribution.sample()
        action_limited = (
            0.5 * (self.act_max - self.act_min) * torch.tanh(action)
            + 0.5 * (self.act_max + self.act_min)
        )
        log_prob = (
            self.gauss_distribution.log_prob(action)
            - torch.sum(torch.log(1 + EPS - torch.pow(torch.tanh(action), 2)), dim=-1)
            - torch.sum(torch.log(0.5 * (self.act_max - self.act_min)), dim=-1)
        ).unsqueeze(-1)
        return action_limited, log_prob

    def rsample(self):
        action = self.gauss_distribution.rsample()
        action_limited = (
            0.5 * (self.act_max - self.act_min) * torch.tanh(action)
            + 0.5 * (self.act_max + self.act_min)
        )
        log_prob = (
            self.gauss_distribution.log_prob(action)
            - torch.sum(torch.log(1 + EPS - torch.pow(torch.tanh(action), 2)), dim=-1)
            - torch.sum(torch.log(0.5 * (self.act_max - self.act_min)), dim=-1)
        ).unsqueeze(-1)
        return action_limited, log_prob

    def log_prob(self, action_limited):
        action = torch.atanh(
            (1 - EPS) * (2 * action_limited - (self.act_max + self.act_min))
            / (self.act_max - self.act_min)
        )
        log_prob = (
            self.gauss_distribution.log_prob(action)
            - torch.sum(torch.log((self.act_max - self.act_min)
            * (1 + EPS - torch.pow(torch.tanh(action), 2))), dim=-1)
        ).unsqueeze(-1)
        return log_prob

    def entropy(self):
        return self.gauss_distribution.entropy()

    def mode(self):
        return (
            0.5 * (self.act_max - self.act_min) * torch.tanh(self.mean)
            + 0.5 * (self.act_max + self.act_min)
        )

    def kl_divergence(self, other):
        return D.kl.kl_divergence(self.gauss_distribution, other.gauss_distribution)