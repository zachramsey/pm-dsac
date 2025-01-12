import sys
import numpy as np
import torch

from preprocess.data_loader import StockDataLoader
from env.trading_env import TradingEnv
from agent.dsac import DSAC

from utils.buffer import ReplayBuffer
from utils.visualize import plot_update_info, animate_portfolio
from utils.metrics import Metrics

class TrainOffPolicy:
    def __init__(self, cfg):
        self.cfg = cfg

        data = StockDataLoader(cfg)
        self.train_dl = data.get_train_data()
        self.eval_dl = data.get_eval_data()

        self.train_dates = self.train_dl.dataset.dates
        self.eval_dates = self.eval_dl.dataset.dates
        
        self.buffer = ReplayBuffer(self.train_dl, self.cfg)
        self.env = TradingEnv(self.cfg)
        self.agent = DSAC(self.cfg)

        self.s = None
        self.weights = np.zeros((self.cfg["train_len"], self.cfg["asset_dim"]))
        self.values = np.zeros(self.cfg["train_len"])

        self.metrics = Metrics(self.cfg)

    def train(self):
        for epoch in range(self.cfg["epochs"]):
            print(f"\nEpoch {epoch}:")
            self._interact(epoch)
            self._update(epoch)
            self._evaluate(epoch)

            if epoch % self.cfg["plot_freq"] == 0 and epoch > 0:
                plot_update_info(epoch, self.agent.update_info, self.cfg)
                #animate_portfolio(epoch, self.weights, self.values, self.train_dates, self.cfg)


    def _interact(self, epoch):
        with torch.no_grad():
            print()
            for step, (feat, targ) in enumerate(self.train_dl):
                feat = feat.squeeze(0)

                if step == 0:
                    self.s = self.env.reset(feat)
                    continue

                a = self.agent.act(self.s)
                r, s_, _ = self.env.step(a, feat, targ)

                if step >= self.cfg["window_size"] - 1:
                    self.buffer.add(epoch, step, a, r)

                self.s = s_
                self.weights[step] = self.env.weights.get_last().flatten().cpu().numpy()
                self.values[step] = self.env.value

                if step % self.cfg["interact_print_freq"] == 0 or step == len(self.train_dl) - 1:
                    sys.stdout.write("\033[F\033[K")
                    print(f"Interact | Step: {step} | Date: {self.train_dates[step].date()} | Value: {self.env.value:.2f}")


    def _update(self, epoch):
        print()
        for step in range(self.cfg["update_steps"]):
            s, a, r, s_ = self.buffer.sample()
            self.agent.update(epoch, s, a, r, s_)
            
            if step % self.cfg["update_print_freq"] == 0 or step == self.cfg["update_steps"] - 1:
                sys.stdout.write("\033[F\033[K")
                print(f"  Update | Step: {step+1} | Actor Loss: {self.agent.update_info["actor_loss"][-1]:.6f} | Critic Loss: {self.agent.update_info["critic_loss"][-1]:.6f}")
        self.agent.log_info(self.cfg["log_dir"] + "latest.log")


    def _evaluate(self, epoch):
        with torch.no_grad():
            self.agent.embedding.eval()

            if epoch % self.cfg["eval_freq"] == 0:
                print()
                for step, (feat, targ) in enumerate(self.eval_dl):
                    feat = feat.squeeze(0)

                    if step == 0:
                        self.s = self.env.reset(feat)
                        continue

                    a = self.agent.act(self.s, is_deterministic=True)
                    r, s_, ret = self.env.step(a, feat, targ)

                    self.s = s_

                    if step % self.cfg["eval_print_freq"] == 0 or step == len(self.eval_dl) - 1:
                        sys.stdout.write("\033[F\033[K")
                        print(f"Evaluate | Step: {step} | Date: {self.eval_dates[step].date()} | Value: {self.env.value:.2f}")
                self.metrics.write(epoch, step, ret, a, self.env.value)
            self.agent.embedding.train()
