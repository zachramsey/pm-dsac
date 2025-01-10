import torch

class ReplayBuffer:
    def __init__(self, train_dl, cfg):
        self.data = train_dl
        self.cfg = cfg
        self.step_offset = 2 * (cfg["window_size"] - 1)
        self.epoch_len = len(train_dl.dataset) - self.step_offset
        self.num_epochs = cfg["capacity"] // self.epoch_len
        self.batch_size = cfg["batch_size"]
        self.asset_dim = cfg["asset_dim"]
        self.window_size = cfg["window_size"]
        self.feat_dim = cfg["feat_dim"]

        self.size = 0

        self.buffer = {
            "i": torch.zeros((self.num_epochs, self.epoch_len, 1)),
            "a": torch.zeros((self.num_epochs, self.epoch_len, self.asset_dim)),
            "r": torch.zeros((self.num_epochs, self.epoch_len, 1))
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
        self.buffer["i"][epoch, step] = torch.tensor(i)             # (1,)
        self.buffer["a"][epoch, step] = a.reshape(self.asset_dim)   # (asset_dim,)
        self.buffer["r"][epoch, step] = r.reshape(1)                # (1,)
        
    def sample(self):
        """ Sample a batch of trajectories from the replay buffer
        Returns:
            s (torch.Tensor): State tensor of shape (asset_dim, window_size, feat_dim)
            a (torch.Tensor): Action tensor of shape (asset_dim, window_size)
            r (torch.Tensor): Reward tensor of shape (1, window_size)
            s_ (torch.Tensor): Next state tensor of shape (asset_dim, window_size, feat_dim)
        """
        epoch = torch.randint(0, self.num_epochs, (self.batch_size,))
        start = torch.randint(0, self.epoch_len - self.window_size - 1, (self.batch_size,))
        end = start + self.window_size

        batch_s = torch.zeros((self.batch_size, self.asset_dim, self.window_size, self.feat_dim))
        batch_a = torch.zeros((self.batch_size, self.asset_dim, self.window_size))
        batch_r = torch.zeros((self.batch_size, 1))
        batch_s_ = torch.zeros((self.batch_size, self.asset_dim, self.window_size, self.feat_dim))

        for i in range(self.batch_size):
            a = self.buffer["a"][epoch, start:end+1]    # (asset_dim, window_size+1)
            r = self.buffer["r"][epoch, end-1]          # (1,)
            
            a = a.transpose(0, 2)                       # (asset_dim, window_size+1, 1)
            r = r.squeeze(1)                            # (1,)
            
            i = self.buffer["i"][epoch, end-1].long()   # Step number
            s = self.data.dataset[i][0]                 # (asset_dim, window_size, feat_dim)
            s_ = self.data.dataset[i+1][0]              # (asset_dim, window_size, feat_dim)

            s[..., -1] = a.squeeze(-1)[:, :-1]          # Replace the last column with the action
            s_[..., -1] = a.squeeze(-1)[:, 1:]          # Replace the last column with the action

            a = a[:, -1]                                # (asset_dim, 1)

            batch_s[i] = s
            batch_a[i] = a
            batch_r[i] = r
            batch_s_[i] = s_

        return batch_s, batch_a, batch_r, batch_s_
    
