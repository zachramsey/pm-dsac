'''
Derived from: github.com/Jingliang-Duan/DSAC-v2/blob/main/dsac_v2.py
'''

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
from copy import deepcopy

# from agent.buffer import Buffer
from agent.value import Critic
from agent.lsre_cann import LSRE_CANN as Actor
from agent.distributions import TanhGaussDistribution

class DSAC:
    def __init__(self, cfg):
        self.gamma = cfg["gamma"]                   # Discount factor
        self.tau = cfg["tau"]                       # Target smoothing coefficient
        self.target_entropy = -cfg["asset_dim"]     # Target entropy
        self.delay_update = cfg["delay_update"]     # Policy update interval
        self.tau_b = cfg.get("tau_b", self.tau)     # Clipping boundary & gradient scalar smoothing coefficient
        self.zeta = cfg["clipping_range"]           # Clipping range
        self.grad_bias = cfg["grad_bias"]           # Avoid grad disapearance in gradient scalar term
        self.std_bias = cfg["std_bias"]             # Avoid gradient explosion in q_std term

        # Networks (Q1, Q2, Pi)
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
        self.log_temperature = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        # Optimizers
        self.critic1_opt = Adam(self.critic1.parameters(), lr=cfg["critic_lr"])
        self.critic2_opt = Adam(self.critic2.parameters(), lr=cfg["critic_lr"])
        self.actor_opt = Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.temperature_opt = Adam([self.log_temperature], lr=cfg["temperature_lr"])

        # Experience replay buffer (contains {})
        # self.buffer = Buffer(**cfg["buffer"])

        # Rolling values
        self.q1_mean_std = -1.0
        self.q2_mean_std = -1.0

        # Collect training information
        self.training_info = {
            "step": [],
            "q1": [],
            "q2": [],
            "q1_std": [],
            "q2_std": [],
            "min_q1_std": [],
            "min_q2_std": [],
            "actor_loss": [],
            "critic_loss": [],
            "policy_mean": [],
            "policy_std": [],
            "entropy": [],
            "alpha": [],
            "q1_mean_std": [],
            "q2_mean_std": [],
        }


    def _actor_objective(self, s, act_new, log_prob_new):
        q1, _ = self.critic1(s, act_new)
        q2, _ = self.critic2(s, act_new)
        actor_loss = torch.mean(torch.exp(self.log_temperature) * log_prob_new - torch.min(q1, q2))
        entropy = -torch.mean(log_prob_new.detach())
        return actor_loss, entropy
    

    def _temperature_objective(self, log_prob_new):
        return -self.log_temperature * torch.mean(log_prob_new.detach() + self.target_entropy)
    

    def _evaluate_critic(self, s, a, r, s_next):
        # Get action for next state
        logits_next_mean, logits_next_std = self.actor_target(s_next)
        a_next, log_prob_a_next = TanhGaussDistribution(logits_next_mean, logits_next_std).rsample()

        # Calculate Q-values for current state
        q1, q1_std = self.critic1(s, a)
        q2, q2_std = self.critic2(s, a)

        # Calculate clipping bounds and gradient scalars for Q-values
        if self.q1_mean_std == -1.0: self.q1_mean_std = torch.mean(q1_std.detach())
        else: self.q1_mean_std = self.tau_b * torch.mean(q1_std.detach()) + (1 - self.tau_b) * self.q1_mean_std
        
        if self.q2_mean_std == -1.0: self.q2_mean_std = torch.mean(q2_std.detach())
        else: self.q2_mean_std = self.tau_b * torch.mean(q2_std.detach()) + (1 - self.tau_b) * self.q2_mean_std

        # Calculate Q-values for next state
        q1_next, q1_next_std = self.critic1_target(s_next, a_next)
        normal = Normal(torch.zeros_like(q1_next), torch.ones_like(q1_next_std))
        z = torch.clamp(normal.sample(), -self.zeta, self.zeta)
        q1_next_sample = q1_next + torch.mul(z, q1_next_std)

        q2_next, q2_next_std = self.critic2_target(s_next, a_next)
        normal = Normal(torch.zeros_like(q2_next), torch.ones_like(q2_next_std))
        z = torch.clamp(normal.sample(), -self.zeta, self.zeta)
        q2_next_sample = q2_next + torch.mul(z, q2_next_std)

        # Calculate target Q-value
        q_next = torch.min(q1_next, q2_next)
        q_next_sample = torch.where(q1_next < q2_next, q1_next_sample, q2_next_sample)

        temperature = torch.exp(self.log_temperature)
        q_target = (r + self.gamma * (q_next - temperature * log_prob_a_next)).detach()
        q_target_sample = (r + self.gamma * (q_next_sample - temperature * log_prob_a_next)).detach()

        # Calculate Critic 1 Loss
        # Mean-related gradient
        q1_std_detach = torch.clamp(q1_std, min=0.0).detach()
        grad_mean = -((q_target - q1).detach() / (torch.pow(q1_std_detach, 2) + self.std_bias)) * q1

        # Std-related gradient
        q1_target_bound = (q1 + torch.clamp(q_target_sample - q1, -self.zeta*q1_std, self.zeta*q1_std)).detach()
        grad_std = -((torch.pow(q1.detach() - q1_target_bound, 2) - torch.pow(q1_std_detach, 2)) 
                    /(torch.pow(q1_std_detach, 3) + self.std_bias)) * q1_std

        q1_loss = (torch.pow(self.q1_mean_std, 2) + self.grad_bias) * torch.mean(grad_mean + grad_std)

        # Calculate Critic 2 Loss
        # Mean-related gradient
        q2_std_detach = torch.clamp(q2_std, min=0.0).detach()
        grad_mean = -((q_target - q2).detach() / (torch.pow(q2_std_detach, 2) + self.std_bias)) * q2

        # Std-related gradient
        q2_target_bound = (q2 + torch.clamp(q_target_sample - q2, -self.zeta*q2_std, self.zeta*q2_std)).detach()
        grad_std = -((torch.pow(q2.detach() - q2_target_bound, 2) - torch.pow(q2_std_detach, 2))
                    /(torch.pow(q2_std_detach, 3) + self.std_bias)) * q2_std
        
        q2_loss = (torch.pow(self.q2_mean_std, 2) + self.grad_bias) * torch.mean(grad_mean + grad_std)

        # Total Critic Loss
        critic_loss = q1_loss + q2_loss

        return (critic_loss, 
                torch.mean(q1.detach()), torch.mean(q2.detach()), 
                torch.mean(q1_std.detach()), torch.mean(q2_std.detach()), 
                torch.min(q1_std).detach(), torch.min(q2_std).detach())
    

    def act(self, s):
        logits_mean, logits_std = self.actor(s)
        act, _ = TanhGaussDistribution(logits_mean, logits_std).rsample()
        return act.detach()


    def update(self, iteration, s, a, r, s_next):
        # Get action for current state
        logits_mean, logits_std = self.actor(s)
        policy_mean = torch.mean(torch.tanh(logits_mean)).item()
        policy_std = torch.mean(logits_std).item()
        act_new, log_prob_new = TanhGaussDistribution(policy_mean, policy_std).rsample()
        self.training_info["act_new"].append(act_new)
        self.training_info["log_prob_new"].append(log_prob_new)

        # Update Critic
        self.critic1_opt.zero_grad()#set_to_none=True)
        self.critic2_opt.zero_grad()
        critic_loss, q1, q2, q1_std, q2_std, q1_min_std, q2_min_std = self._evaluate_critic(s, a, r, s_next)
        critic_loss.backward()

        # Don't consider critic when updating actor
        for p in self.critic1.parameters(): p.requires_grad = False
        for p in self.critic2.parameters(): p.requires_grad = False

        # Update Actor
        self.actor_opt.zero_grad()
        actor_loss, entropy = self._actor_objective(s, act_new, log_prob_new)
        actor_loss.backward()

        # Re-enable critic
        for p in self.critic1.parameters(): p.requires_grad = True
        for p in self.critic2.parameters(): p.requires_grad = True

        # Update temperature
        self.temperature_opt.zero_grad()
        temperature_loss = self._temperature_objective(log_prob_new)
        temperature_loss.backward()

        # Collect training information
        self.training_info["step"].append(iteration)
        self.training_info["q1"].append(q1.item())
        self.training_info["q2"].append(q2.item())
        self.training_info["q1_std"].append(q1_std.item())
        self.training_info["q2_std"].append(q2_std.item())
        self.training_info["min_q1_std"].append(q1_min_std.item())
        self.training_info["min_q2_std"].append(q2_min_std.item())
        self.training_info["actor_loss"].append(actor_loss.item())
        self.training_info["critic_loss"].append(critic_loss.item())
        self.training_info["policy_mean"].append(policy_mean)
        self.training_info["policy_std"].append(policy_std)
        self.training_info["entropy"].append(entropy.item())
        self.training_info["alpha"].append(torch.exp(self.log_temperature).item())
        self.training_info["q1_mean_std"].append(self.q1_mean_std)
        self.training_info["q2_mean_std"].append(self.q2_mean_std)

        # Optimize critic networks
        self.critic1_opt.step()
        self.critic2_opt.step()

        # Delayed update
        if iteration % self.delay_update == 0:
            self.actor_opt.step()       # Optimize actor network
            self.temperature_opt.step() # Optimize temperature network

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


    def log_training_info(self, log_file):
        # Check that logs directory exists
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Write training information to log file
        with open(log_file, "a") as f:
            f.write("="*50 + "\n")
            for key, value in self.training_info.items():
                if key == "step":
                    f.write(f"Training Step: {value[-1]}\n")
                    f.write("-"*50 + "\n")
                else:
                    f.write(f"{key}: {value[-1]}\n")
            f.write("-"*50 + "\n\n")