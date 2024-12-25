import torch

class ReplayBuffer:
    def __init__(self, train_dl, cfg):
        self.data = train_dl
        self.cfg = cfg
        self.step_offset = 2 * (cfg["window_size"] - 1)
        self.epoch_len = len(train_dl.dataset) - self.step_offset
        self.num_epochs = cfg["capacity"] // self.epoch_len
        self.asset_dim = cfg["asset_dim"]
        self.window_size = cfg["window_size"]

        self.buffer = {
            "i": torch.zeros((self.num_epochs, self.epoch_len, 1)),
            "a": torch.zeros((self.num_epochs, self.epoch_len, self.asset_dim)),
            "r": torch.zeros((self.num_epochs, self.epoch_len, 1)),
        }

    def add(self, e, i, a, r):
        """ Add an experience to the replay buffer
        Args:
            e (int): Epoch number
            i (int): Step number
            a (torch.Tensor): Action tensor of shape (asset_dim, 1)
            r (torch.Tensor): Reward tensor of shape (1,)
        """
        epoch = e % self.num_epochs
        step = i - self.step_offset
        self.buffer["i"][epoch, step] = torch.tensor(i)
        self.buffer["a"][epoch, step] = a.reshape(self.asset_dim)
        self.buffer["r"][epoch, step] = r
        
    def sample(self):
        """ Sample a batch of trajectories from the replay buffer
        Returns:
            s (torch.Tensor): State tensor of shape (asset_dim, window_size, feat_dim)
            a (torch.Tensor): Action tensor of shape (asset_dim, window_size)
            r (torch.Tensor): Reward tensor of shape (1, window_size)
            s_ (torch.Tensor): Next state tensor of shape (asset_dim, window_size, feat_dim)
        """
        epoch = torch.randint(0, self.num_epochs, (1,))
        start = torch.randint(0, self.epoch_len - self.cfg["window_size"] - 1, (1,))
        end = start + self.cfg["window_size"]
        
        i = self.buffer["i"][epoch, end-1].long()   # Step number

        s = self.data.dataset[i]                    # (asset_dim, window_size, feat_dim)
        a = self.buffer["a"][epoch, start:end+1]    # (window_size, asset_dim)
        r = self.buffer["r"][epoch, start:end]      # (window_size,)
        s_ = self.data.dataset[i+1]                 # (asset_dim, window_size, feat_dim)

        a = a.reshape(self.asset_dim, -1)           # (asset_dim, window_size)
        r = r.unsqueeze(0)                          # (1, window_size)

        s[..., -1] = a[:, :-1]                      # Replace the last column with the action
        s_[..., -1] = a[:, 1:]                      # Replace the last column with the action

        return s, a, r, s_