import yaml
from data_loader import StockData
from environment import TradingEnv
from dsac import DSAC

class StockTrader:
    def __init__(self, cfg):
        self.cfg = cfg

        data_set = StockData(cfg)
        self.train_dl, self.eval_dl, self.cfg = data_set.get_data_loaders()
        
        self.env = TradingEnv(cfg)
        self.agent = DSAC(cfg)

        self.state = None
        
    def train(self):
        for epoch in range(self.cfg["epochs"]):
            for step, (date, features, targets) in enumerate(self.train_dl):
                if step == 0:
                    self.state = self.env.reset(features)
                else:
                    actions = self.agent.act(self.state)
                    next_state, reward = self.env.step(actions, features, targets)
                    self.agent.update(self.state, actions, reward, next_state)
                    self.state = next_state

    def eval(self):
        for step, (date, features, targets) in enumerate(self.eval_dl):
            if step == 0:
                self.state = self.env.reset(features)
            else:
                actions = self.agent.act(self.state)
                next_state, reward = self.env.step(actions, features, targets)
                self.state = next_state


def main():
    cfg_file = "configs/base.yaml"
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    trader = StockTrader(cfg)
    trader.train()

if __name__ == "__main__":
    main()
