import torch
import torch.nn.functional as F

import numpy as np

class TradingEnv:
    def __init__(self, cfg):
        """Initialize the Trading Environment
        Args:
            cfg (dict): Configuration dictionary
        """
        self.init_cash = cfg["init_cash"]
        self.rew_scale = cfg["reward_scale"]
        self.num_stocks = cfg["num_stocks"]
        self.num_features = cfg["num_features"]
        self.transaction_cost = cfg["transaction_cost"]

        self.value_prev = None
        self.actions_prev = None
        self.returns = None


    def reset(self, features):
        """Reset the environment to the initial state
        Args:
            features (Tensor [num_stocks, num_features]): Features for the first day of trading
        Returns:
            state (Tensor [num_stocks, num_features + 1]): Features for the first day of trading after actions appended
        """
        self.value_prev = self.init_cash
        self.actions_prev = [1] + [0] * self.num_stocks-1
        self.returns = []
        return self._get_state(features, self.actions_prev)


    def step(self, actions, features, targets):
        """Take a step in the environment based on the given actions
        Args:
            actions (Tensor [num_stocks]): stock weights to updating the portfolio
            features (Tensor [num_stocks, num_features]): Features for the current day 
            targets (Tensor [num_stocks]): Target stock prices for the current day
        Returns:
            state_next (Tensor [num_stocks, num_features + 1]): Features for the next day
            reward (float): Reward for the current step
        """
        # Calculate the value after considering transaction costs
        value = self._value_after_cost(self.actions_prev, actions, self.transaction_cost)
        
        # Calculate the log returns for the current day
        self.returns.append(np.log(value / self.value_prev))

        # Update the previous actions and value
        self.actions_prev = actions
        self.value_prev = value
        
        # Combine the features and actions to get the state for the next day
        state_next = torch.cat([features, actions.view(-1, 1)], dim=1)

        # Calculate the reward for the current day
        reward = self.returns[-1] * self.rew_scale

        return state_next, reward


    # derived from https://github.com/ZhengyaoJiang/PGPortfolio/blob/master/pgportfolio/tools/trade.py
    def _value_after_cost(self, w0, w1, c):
        """Calculate the value after trading with transaction costs
        Args:
            w0 (Tensor [num_stocks]): Weights of the stocks before trading
            w1 (Tensor [num_stocks]): Weights of the stocks after trading
            c (float): Transaction cost
        Returns:
            v1 (float): Value after trading with transaction costs
        """
        v0 = 1
        v1 = 1 - 2*c + c**2
        while abs(v1 - v0) > 1e-10:
            v0 = v1
            v1 = (1 - c*w0[0] - (2*c - c**2) * np.sum(np.maximum(w0[1:] - v1*w1[1:], 0))) / (1 - c*w1[0])
        return v1
    