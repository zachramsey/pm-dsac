import torch
from preprocess.data_loader import StockDataLoader
from buffer import ReplayBuffer
from env.trading_env import TradingEnv
from agent.dsac import DSAC

class TrainOffPolicy:
    def __init__(self, cfg):
        self.cfg = cfg

        data = StockDataLoader(cfg)
        self.train_dl = data.get_train_data()
        self.eval_dl = data.get_test_data()

        self.train_dates = self.train_dl.dataset.dates
        self.eval_dates = self.eval_dl.dataset.dates
        
        self.buffer = ReplayBuffer(self.train_dl, self.cfg)
        self.env = TradingEnv(self.cfg)
        self.agent = DSAC(self.cfg)

        self.state = None
        
    def train(self):
        for epoch in range(self.cfg["epochs"]):

            # Environment Interaction
            print(f"\nEpoch {epoch} Training")
            with torch.no_grad():
                for step, (features, targets) in enumerate(self.train_dl):
                    features = features.squeeze(0)

                    if step == 0:
                        self.state = self.env.reset(features)
                        continue

                    action = self.agent.act(self.state)
                    reward, next_state = self.env.step(action, features, targets)

                    if step >= self.cfg["window_size"] - 1:
                        self.buffer.add(epoch, step, action, reward)

                    self.state = next_state

                    if step % self.cfg["print_freq"] == 0:
                        print(f"Epoch: {epoch} | Step: {step} | Date: {self.train_dates[step].date()} | Reward: {reward:.6f}")

            # Agent Update
            for step in range(self.cfg["update_steps"]):
                s, a, r, s_ = self.buffer.sample()
                self.agent.update(epoch, s, a, r, s_)
                self.agent.log_training_info(self.cfg["log_dir"] + "latest.log")

            # Evaluation
            print(f"\nEpoch {epoch} Evaluation")
            with torch.no_grad():
                if epoch % self.cfg["eval_freq"] == 0:
                    for step, (features, targets) in enumerate(self.eval_dl):
                        features = features.squeeze(0)

                        if step == 0:
                            self.state = self.env.reset(features)
                            continue

                        action = self.agent.act(self.state)
                        next_state, reward = self.env.step(action, features, targets)

                        if step % self.cfg["print_freq"] == 0:
                            print(f"Epoch: {epoch} | Step: {step} | Date: {self.eval_dates[step].date()} | Reward: {reward:.6f}")