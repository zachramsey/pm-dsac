import yaml
from datetime import datetime as dt
from train import TrainOffPolicy

if __name__ == "__main__":

    # Load config file
    cfg_file = "configs/base.yaml"
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    # Clear the latest log file
    with open(cfg["log_dir"] + "latest.log", "w") as f:
        f.write("")

    try:
        # Off-Policy Trainingactor_loss = torch.mean(torch.exp(self.log_alpha) * log_prob_new - q_min + std_min)
        trader = TrainOffPolicy(cfg)
        trader.train()
    except KeyboardInterrupt:
        # Save the latest log file
        datetime_str = dt.now().strftime("%y-%m-%d_%H%M%S")
        with open(cfg["log_dir"] + "latest.log", "r") as rf:
            with open(cfg["log_dir"] + f"{datetime_str}.log", "w") as wf:
                wf.write(rf.read())