'''
Derived from: github.com/Jingliang-Duan/DSAC-v2/blob/main/dsac_v2.py
'''

import os
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal

# from agent.buffer import Buffer
from agent.lsre_cann import LSRE_CANN
from agent.value import Critic
from agent.policy import Actor
from utils.distributions import TanhGaussDistribution

class DSAC:
    def __init__(self, cfg):
        self.asset_dim = cfg["asset_dim"]           # Number of assets

        self.gamma = cfg["gamma"]                   # Discount factor
        self.tau = cfg["tau"]                       # Target smoothing coefficient
        self.target_entropy = -cfg["asset_dim"]     # Target entropy
        self.delay_update = cfg["delay_update"]     # Policy update interval
        self.tau_b = cfg.get("tau_b", self.tau)     # Clipping boundary & gradient scalar smoothing coefficient
        self.zeta = cfg["clipping_range"]           # Clipping range
        self.grad_bias = cfg["grad_bias"]           # Avoid grad disapearance in gradient scalar term
        self.std_bias = cfg["std_bias"]             # Avoid gradient explosion in q_std term

        # Networks (Q1, Q2, Pi)
        self.embedding = LSRE_CANN(cfg)
        self.critic1 = Critic(cfg)
        self.critic2 = Critic(cfg)
        self.actor = Actor(cfg)

        # Target Networks
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)
        self.actor_target = deepcopy(self.actor)

        # Perform soft update (Polyak update) on target networks
        for p in self.critic1_target.parameters(): p.requires_grad = False
        for p in self.critic2_target.parameters(): p.requires_grad = False
        for p in self.actor_target.parameters(): p.requires_grad = False

        # Parameterized policy entropy temperature (Alpha)
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        # Optimizers
        self.embedding_opt = Adam(self.embedding.parameters(), lr=cfg["embedding_lr"])
        self.critic1_opt = Adam(self.critic1.parameters(), lr=cfg["critic_lr"])
        self.critic2_opt = Adam(self.critic2.parameters(), lr=cfg["critic_lr"])
        self.actor_opt = Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.alpha_opt = Adam([self.log_alpha], lr=cfg["temperature_lr"])

        # Moving Averages
        self.q1_std_bound = 0.0
        self.q2_std_bound = 0.0
        self.q1_grad_scalar = 0.0
        self.q2_grad_scalar = 0.0

        # Collect training information
        self.update_info = {
            "step": [],
            "q1": [],
            "q2": [],
            "q1_std": [],
            "q2_std": [],
            "q1_mean_std": [],
            "q2_mean_std": [],
            "q1_loss": [],
            "q2_loss": [],
            "critic_loss": [],
            "actor_loss": [],
            "entropy": [],
            "alpha": [],
        }


    def reset(self):
        self.embedding.reset_latent()


    def act(self, s, is_deterministic=False):
        h, z = self.embedding(s)
        act, std = self.actor(h)
        act_dist = TanhGaussDistribution(act, std)
        act = act_dist.mode() if is_deterministic else act_dist.sample()[0]
        act = torch.softmax(act, dim=0)
        return act.detach(), z.detach()
    

    def _critic_objective(self, s, a, r, s_next):
        # Calculate Q-values for current state
        q1, q1_std = self.critic1(s, a)
        q2, q2_std = self.critic2(s, a)

        # Calculate clipping bounds and gradient scalars
        self.q1_std_bound = self.tau_b * self.zeta * torch.mean(q1_std.detach()) + (1 - self.tau_b) * self.q1_std_bound
        self.q2_std_bound = self.tau_b * self.zeta * torch.mean(q2_std.detach()) + (1 - self.tau_b) * self.q2_std_bound
        
        self.q1_grad_scalar = self.tau_b * torch.mean(torch.pow(q1_std.detach(), 2)) + (1 - self.tau_b) * self.q1_grad_scalar
        self.q2_grad_scalar = self.tau_b * torch.mean(torch.pow(q2_std.detach(), 2)) + (1 - self.tau_b) * self.q2_grad_scalar

        # Get action for next state from target policy
        logits_next_mean, logits_next_std = self.actor_target(s_next)
        a_next, log_prob_a_next = TanhGaussDistribution(logits_next_mean, logits_next_std).rsample()
        a_next = torch.softmax(a_next, dim=0)

        # Determine minimum Q-value function
        q1_next, q1_next_std = self.critic1_target(s_next, a_next)
        q2_next, q2_next_std = self.critic2_target(s_next, a_next)
        q_next = torch.min(q1_next, q2_next)
        q_next_std = torch.where(q1_next > q2_next, q1_next_std, q2_next_std)

        # Entropy temperature
        alpha = torch.exp(self.log_alpha)

        # Target Q-value
        q_targ = (r + self.gamma * (q_next - alpha * log_prob_a_next)).detach()

        # Target returns
        z = Normal(q_next, torch.pow(q_next_std, 2)).sample()
        ret_targ = (r + self.gamma * (z - alpha * log_prob_a_next)).detach()

        # Critic 1 Loss
        std_detach = torch.clamp(q1_std, min=0.0).detach()
        grad_mean = -((q_targ - q1).detach() / (torch.pow(std_detach, 2) + self.std_bias)) * q1
        
        ret_targ_bound = torch.clamp(ret_targ, q1-self.q1_std_bound, q1+self.q1_std_bound).detach()
        grad_std = -((torch.pow(q1.detach() - ret_targ_bound, 2) - torch.pow(std_detach, 2)) 
                    /(torch.pow(std_detach, 3) + self.std_bias)) * q1_std
        
        q1_loss = (self.q1_grad_scalar + self.grad_bias) * torch.mean(grad_mean + grad_std)

        # Critic 2 Loss
        std_detach = torch.clamp(q2_std, min=0.0).detach()
        grad_mean = -((q_targ - q2).detach() / (torch.pow(std_detach, 2) + self.std_bias)) * q2
        
        ret_targ_bound = torch.clamp(ret_targ, q2-self.q2_std_bound, q2+self.q2_std_bound).detach()
        grad_std = -((torch.pow(q2.detach() - ret_targ_bound, 2) - torch.pow(std_detach, 2)) 
                    /(torch.pow(std_detach, 3) + self.std_bias)) * q2_std
        
        q2_loss = (self.q2_grad_scalar + self.grad_bias) * torch.mean(grad_mean + grad_std)

        return (q1_loss+q2_loss, q1_loss.detach(), q2_loss.detach(), 
                torch.mean(q1.detach()), torch.mean(q2.detach()), 
                torch.mean(q1_std.detach()), torch.mean(q2_std.detach()))
    

    def _actor_objective(self, s, act_new, log_prob_new):
        q1, _ = self.critic1(s, act_new)
        q2, _ = self.critic2(s, act_new)
        actor_loss = torch.mean(torch.exp(self.log_alpha) * log_prob_new - torch.min(q1, q2))
        entropy = -torch.mean(log_prob_new.detach())
        return actor_loss, entropy


    def _temperature_objective(self, log_prob_new):
        return -self.log_alpha * torch.mean(log_prob_new.detach() + self.target_entropy)


    def update(self, iteration, s, z, a, r, s_next):
        # Embed state into latent space
        self.embedding_opt.zero_grad()
        h, _ = self.embedding(s, z)
        h_next, _ = self.embedding(s_next)
        
        # Get action for current state
        logits_mean, logits_std = self.actor(h)
        act_new, log_prob_new = TanhGaussDistribution(logits_mean, logits_std).rsample()
        act_new = torch.softmax(act_new, dim=0)

        # Update Critic
        self.critic1_opt.zero_grad()#set_to_none=True)
        self.critic2_opt.zero_grad()
        q_loss, q1_loss, q2_loss, q1, q2, q1_std, q2_std = self._critic_objective(h, a, r, h_next)
        q_loss.backward(retain_graph=True)

        # Don't consider critic when updating actor
        for p in self.critic1.parameters(): p.requires_grad = False
        for p in self.critic2.parameters(): p.requires_grad = False

        # Update Actor
        self.actor_opt.zero_grad()
        actor_loss, entropy = self._actor_objective(h, act_new, log_prob_new)
        actor_loss.backward()

        # Re-enable critic
        for p in self.critic1.parameters(): p.requires_grad = True
        for p in self.critic2.parameters(): p.requires_grad = True

        # Update temperature
        self.alpha_opt.zero_grad()
        temperature_loss = self._temperature_objective(log_prob_new)
        temperature_loss.backward()

        # Optimize critic networks
        self.embedding_opt.step()
        self.critic1_opt.step()
        self.critic2_opt.step()

        # Delayed update
        if iteration % self.delay_update == 0:
            self.actor_opt.step()       # Optimize actor network
            self.alpha_opt.step() # Optimize temperature network

            # Perform soft update on target networks
            with torch.no_grad():
                polyak = 1 - self.tau

                for p, p_targ in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

                for p, p_targ in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

                for p, p_targ in zip(self.actor.parameters(), self.actor_target.parameters(),):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

        # Collect training information
        self.update_info["step"].append(iteration)
        self.update_info["q1"].append(q1.item())
        self.update_info["q2"].append(q2.item())
        self.update_info["q1_std"].append(q1_std.item())
        self.update_info["q2_std"].append(q2_std.item())
        self.update_info["q1_mean_std"].append(torch.sqrt(self.q1_grad_scalar).item())
        self.update_info["q2_mean_std"].append(torch.sqrt(self.q2_grad_scalar).item())
        self.update_info["q1_loss"].append(q1_loss.item())
        self.update_info["q2_loss"].append(q2_loss.item())
        self.update_info["critic_loss"].append(q_loss.item())
        self.update_info["actor_loss"].append(actor_loss.item())
        self.update_info["entropy"].append(entropy.item())
        self.update_info["alpha"].append(torch.exp(self.log_alpha).item())

    def log_info(self, log_file):
        # Check that logs directory exists
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Write training information to log file
        with open(log_file, "a") as f:
            f.write("="*50 + "\n")
            for key, value in self.update_info.items():
                if key == "step":
                    f.write(f"Training Step: {value[-1]}\n")
                    f.write("-"*50 + "\n")
                else:
                    f.write(f"{key}: {value[-1]}\n")
            f.write("-"*50 + "\n\n")
