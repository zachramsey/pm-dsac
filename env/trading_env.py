import torch

from env.weight_buffer import ActionBuffer

class TradingEnv:
    def __init__(self, cfg):
        self.init_cash = cfg["init_cash"]
        self.c_sell = cfg["sell_cost"]
        self.c_buy = cfg["purchase_cost"]

        self.value = self.init_cash         # Initialize the portfolio value
        self.weights = ActionBuffer(cfg)    # Buffer for storing the weights


    # Derived from: github.com/ZhengyaoJiang/PGPortfolio/blob/master/pgportfolio/tools/trade.py
    def _calc_remainder(self, w, w_targ):
        """ Calculate the transaction remainder factor -- the percentage of the portfolio that remains after transaction fees.
        Args:
            w (torch.Tensor): Current portfolio weights
            w_targ (torch.Tensor): Desired new portfolio weights
        Returns:
            float: Transaction remainder factor
        """
        mu0 = torch.tensor(1.0, device=w.device)
        mu1 = torch.tensor(1.0 - self.c_sell - self.c_buy + self.c_sell * self.c_buy, device=w.device)

        while torch.abs(mu1 - mu0) > 1e-10:
            mu0 = mu1
            denom = 1 - self.c_buy * w_targ[0]
            coeff = 1 - (self.c_buy * w[0]) - (self.c_sell + self.c_buy - (self.c_sell * self.c_buy))
            mu1 = (coeff * torch.sum(torch.maximum(w[1:] - mu1 * w_targ[1:], torch.tensor(0.0, device=w.device)))) / denom

        return mu1.item()


    def step(self, action, features, rel_prices):
        """ Execute a trading action and return new state, reward, and other info.
        Args:
            action (torch.Tensor): Desired portfolio weights
            features (torch.Tensor): Features for the current day
            rel_prices (torch.Tensor): Price relative vector for the current day
        Returns:
            features (torch.Tensor): Updated features for the next day
            reward (float): Reward for the current day
        """
        action = action.flatten()                       # Flatten the action tensor
        # action = torch.softmax(action, dim=-1)          # Apply softmax to the action tensor
        rel_prices = rel_prices.flatten()               # Flatten the price relative tensor

        w = self.weights.get_last()                     # Get the portfolio weights before the action

        # # Calculate the transaction remainder factor
        # mu = self._calc_remainder(w, action)

        # Calculate the portfolio value after the transaction
        value = self.value * torch.dot(rel_prices, action)

        # Calculate the reward for the current day
        reward = torch.log(value / self.value)
        self.value = value

        # Update the weights buffer with the new weights
        self.weights.update(action)

        # Replace the last column of the features with the weights
        features[:, :, -1] = self.weights.get_all()

        return reward, features


    def reset(self, features):
        """ Reset the environment to the initial state.
        Args:
            features (torch.Tensor): Features for the first day
        Returns:
            features (torch.Tensor): Updated features for the first day
        """
        self.value = self.init_cash     # Reset the portfolio value
        self.weights.reset()            # Reset the weights buffer

        action = self.weights.get_all() # Get the action from the buffer
        features[:, :, -1] = action     # Replace the last column with the weights

        return features
    