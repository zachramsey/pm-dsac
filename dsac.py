import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
from copy import deepcopy

from buffer import Buffer
from critic import Critic
from actor import Actor
from distributions import TanhGaussDistribution

class DSAC:
    def __init__(self, **cfg):
        self.gamma = cfg["gamma"]                   # Discount factor
        self.tau = cfg["tau"]                       # Target smoothing coefficient
        self.alpha = cfg.get("alpha", 0.2)          # Initial entropy temperature
        self.target_entropy = cfg["target_entropy"] # TODO ???
        self.delay_update = cfg["delay_update"]     # Policy update interval
        self.tau_b = cfg.get("tau_b", self.tau)     # Clipping boundary & gradient scalar smoothing coefficient
        self.zeta = cfg["zeta"]                     # Clipping range
        self.bias = cfg["bias"]                     # For eps & eps_w to prevent grad explosion and disappearance

        # Networks (Q1, Q2, Pi)
        self.critic1 = Critic(**cfg["critic"])
        self.critic2 = Critic(**cfg["critic"])
        self.actor = Actor(**cfg["actor"])

        # Target Networks
        self.critic_1_target = deepcopy(self.critic1)
        self.critic_2_target = deepcopy(self.critic2)
        self.actor_target = deepcopy(self.actor)

        # Perform soft update (Polyak update) on target networks
        for p in self.critic_1_target.parameters(): p.requires_grad = False
        for p in self.critic_2_target.parameters(): p.requires_grad = False
        for p in self.actor_target.parameters(): p.requires_grad = False

        # Parameterized policy entropy temperature (Alpha)
        self.log_temperature = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        # Optimizers
        self.critic_1_opt = Adam(self.critic1.parameters(), lr=cfg["critic1_lr"])
        self.critic_2_opt = Adam(self.critic2.parameters(), lr=cfg["critic2_lr"])
        self.actor_opt = Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.temperature_opt = Adam([self.log_temperature], lr=cfg["temperature_lr"])

        # Experience replay buffer (contains {})
        self.buffer = Buffer(**cfg["buffer"])

        # Rolling values
        self.q1_bound = None
        self.q2_bound = None
        self.q1_grad_scalar = None
        self.q2_grad_scalar = None


    def calc_critic_loss(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        # Get action for next state
        logits_next_mean, logits_next_std = self.actor_target(s_next)
        a_next_dist = TanhGaussDistribution(logits_next_mean, logits_next_std)
        a_next, log_prob_a_next = a_next_dist.rsample()

        # Calculate Q-values for current state
        q1, q1_std = self.critic1(s, a)
        q2, q2_std = self.critic2(s, a)

        # Calculate clipping bounds and gradient scalars for Q-values
        mean_std_1 = torch.mean(q1_std.detach())
        self.q1_bound = self.tau_b * self.zeta * mean_std_1 + (1 - self.tau_b) * self.q1_bound
        self.q1_grad_scalar = self.tau_b * torch.pow(mean_std_1, 2) + (1 - self.tau_b) * self.q1_bound

        mean_std_2 = torch.mean(q2_std.detach())
        self.q2_bound = self.tau_b * self.zeta * mean_std_2 + (1 - self.tau_b) * self.q2_bound
        self.q2_grad_scalar = self.tau_b * torch.pow(mean_std_2, 2) + (1 - self.tau_b) * self.q2_bound

        # Calculate Q-values for next state
        q1_next, q1_next_std = self.critic1(s_next, a_next)
        normal = Normal(torch.zeros_like(q1_next), torch.ones_like(q1_next_std))
        z = torch.clamp(normal.sample(), -3, 3)
        q1_next_sample = q1_next + torch.mul(z, q1_next_std)
        
        q2_next, q2_next_std = self.critic2(s_next, a_next)
        normal = Normal(torch.zeros_like(q2_next), torch.ones_like(q2_next_std))
        z = torch.clamp(normal.sample(), -3, 3)
        q2_next_sample = q2_next + torch.mul(z, q2_next_std)

        # Calculate target Q-value
        q_next = torch.min(q1_next, q2_next)
        q_next_sample = torch.where(q1_next < q2_next, q1_next_sample, q2_next_sample)

        temperature = torch.exp(self.log_temperature)
        q_target = r + self.gamma * (q_next - temperature * log_prob_a_next)
        q_target_sample = r + self.gamma * (q_next_sample - temperature * log_prob_a_next)

        # Calculate q1 loss
        q1_target_bound = q1 + torch.clamp(q_target_sample - q1, -self.q1_bound, self.q1_bound)
        q1_std_detach = torch.clamp(q1_std, min=0.0).detach()
        q1_loss = (self.q1_grad_scalar + self.bias) * torch.mean(
            -(q_target - q1).detach() / (torch.pow(q1_std_detach, 2) + self.bias) * q1
            -((torch.pow(q1_target_bound - q1.detach(), 2)- torch.pow(q1_std_detach, 2)) 
              / (torch.pow(q1_std_detach, 3) + self.bias)) * q1_std
        )

        # Calculate q2 loss
        q2_std_detach = torch.clamp(q2_std, min=0.0).detach()
        q2_target_bound = q1 + torch.clamp(q_target_sample - q2, -self.q2_bound, self.q2_bound)
        q2_loss = (self.q2_grad_scalar + self.bias) * torch.mean(
            -(q_target - q2).detach() / (torch.pow(q2_std_detach, 2) + self.bias) * q2
            -((torch.pow(q2_target_bound - q2.detach(), 2)- torch.pow(q2_std_detach, 2)) 
              / (torch.pow(q2_std_detach, 3) + self.bias)) * q2_std
        )

        return (q1_loss + q2_loss, 
                torch.mean(q1.detach()), torch.mean(q2.detach()), 
                torch.mean(q1_std.detach()), torch.mean(q2_std.detach()), 
                torch.min(q1_std).detach(), torch.min(q2_std).detach())


    def calc_actor_loss(self, s: torch.Tensor, act_new: torch.Tensor, log_prob_new: torch.Tensor) -> torch.Tensor:
        q1, _ = self.critic1(s, act_new)
        q2, _ = self.critic2(s, act_new)
        actor_loss = torch.mean(torch.exp(self.log_temperature) * log_prob_new - torch.min(q1,q2))
        entropy = -torch.mean(log_prob_new.detach())
        return actor_loss, entropy
        

    def calc_entropy_loss(self, log_prob_new: torch.Tensor) -> torch.Tensor:
        return -self.log_temperature * torch.mean(log_prob_new.detach() + self.target_entropy)


    def update(self, iteration: int) -> None:
        data = self.buffer.sample()
        s, a, r, s_next = data["obs"], data["act"], data["rew"], data["obs_next"]

        # Get action for current state
        logits_mean, logits_std = self.actor(s)
        act_mean = torch.tanh(logits_mean)
        act_std = torch.mean(logits_std).item()
        act_dist = TanhGaussDistribution(act_mean, act_std)
        act_new, log_prob_new = act_dist.rsample()

        # Calculate critic gradient
        self.critic_1_opt.zero_grad()#set_to_none=True)
        self.critic_2_opt.zero_grad()
        critic_loss, q1, q2, q1_std, q2_std, q1_min_std, q2_min_std = self.calc_critic_loss(s, a, r, s_next)
        critic_loss.backward()

        # Don't consider critic when calculating actor gradient
        for p in self.critic1.parameters(): p.requires_grad = False
        for p in self.critic2.parameters(): p.requires_grad = False

        # Calculate actor gradient
        self.actor_opt.zero_grad()
        actor_loss, entropy = self.calc_actor_loss(s, act_new, log_prob_new)
        actor_loss.backward()

        # Re-enable critic gradient
        for p in self.critic1.parameters(): p.requires_grad = True
        for p in self.critic2.parameters(): p.requires_grad = True

        # Calculate entropy temperature gradient
        self.temperature_opt.zero_grad()
        entropy_loss = self.calc_entropy_loss(log_prob_new)
        entropy_loss.backward()

        # Optimize critic networks
        self.critic_1_opt.step()
        self.critic_2_opt.step()

        # Delayed update
        if iteration % self.delay_update == 0:
            self.actor_opt.step()       # Optimize actor network
            self.temperature_opt.step() # Optimize temperature network

            # Perform soft update on target networks
            with torch.no_grad():
                polyak = 1 - self.tau

                for p, p_targ in zip(self.critic1.parameters(), self.critic_1_target.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

                for p, p_targ in zip(self.critic2.parameters(), self.critic_2_target.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

                for p, p_targ in zip(self.actor.parameters(), self.actor_target.parameters(),):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
