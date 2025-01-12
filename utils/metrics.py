
import numpy as np

class Metrics:
    def __init__(self, cfg):
        self.log_file = cfg["log_dir"] + "latest.log"
        self.returns = np.zeros((cfg["eval_len"]))
        self.weights = np.zeros((cfg["eval_len"], cfg["asset_dim"]))

    def sharpe(self):
        ''' ### Calculate Annualized Sharpe ratio
        Args:
            returns (np.ndarray): Returns of the portfolio
        Returns:
            sharpe (float): Sharpe ratio
        '''
        return np.sqrt(252) * np.mean(self.returns) / np.std(self.returns)

    def mdd(self):
        ''' ### Calculate Maximum Drawdown
        Args:
            returns (np.ndarray): Returns of the portfolio
        Returns:
            mdd (float): Maximum Draw
        '''
        cum_returns = np.cumprod(1 + self.returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / peak
        return np.min(drawdown)

    def average_turnover(self):
        ''' ### Calculate Average Turnover
        Args:
            weights (np.ndarray): Weights of the portfolio
        Returns:
            avg_turnover (float): Average Turnover
        '''
        return np.sum(np.abs(self.weights[1:] - self.weights[:-1])) / (2 * (len(self.weights) - 1)) * 100
    
    def write(self, epoch, step, returns, weights, value):
        ''' ### Write metrics to the log file
        Args:
            epoch (int): Current epoch
            returns (np.ndarray): Returns of the portfolio
            weights (np.ndarray): Weights of the portfolio
            values (np.ndarray): Portfolio values
            cfg (dict): Configuration dictionary
        '''
        self.returns[step] = returns
        self.weights[step] = weights.flatten()

        with open(self.log_file, "a") as f:
            f.write(f"Evaluation Step:\n")
            f.write(f"     Sharpe Ratio: {self.sharpe()}\n")
            f.write(f"Maximum Draw-Down: {self.mdd()}\n")
            f.write(f" Average Turnover: {self.average_turnover()}\n")
            f.write(f"      Final Value: {value:.2f}\n\n")
