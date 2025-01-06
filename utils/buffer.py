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
            "z": torch.zeros((self.num_epochs, self.epoch_len, self.asset_dim, cfg["num_latents"], cfg["latent_dim"])),
            "a": torch.zeros((self.num_epochs, self.epoch_len, self.asset_dim)),
            "r": torch.zeros((self.num_epochs, self.epoch_len, 1))
        }

    def add(self, e, i, z, a, r):
        """ Add an experience to the replay buffer
        Args:
            e (int): Epoch number
            i (int): Step number
            z (torch.Tensor): Latent state tensor of shape (asset_dim, num_latents, latent_dim)
            a (torch.Tensor): Action tensor of shape (asset_dim, 1)
            r (torch.Tensor): Reward tensor of shape (1,)
        """
        epoch = e % self.num_epochs
        step = i - self.step_offset
        self.buffer["i"][epoch, step] = torch.tensor(i)             # (1,)
        self.buffer["z"][epoch, step] = z                           # (asset_dim, num_latents, latent_dim)
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
        epoch = torch.randint(0, self.num_epochs, (1,))
        start = torch.randint(0, self.epoch_len - self.window_size - 1, (1,))
        end = start + self.window_size
        
        i = self.buffer["i"][epoch, end-1].long()   # Step number

        s = self.data.dataset[i][0]                 # (asset_dim, window_size, feat_dim)
        z = self.buffer["z"][epoch, end-1]          # (asset_dim, num_latents, latent_dim)
        a = self.buffer["a"][epoch, start:end+1]    # (asset_dim, window_size+1)
        r = self.buffer["r"][epoch, end-1]          # (1,)
        s_ = self.data.dataset[i+1][0]              # (asset_dim, window_size, feat_dim)

        z = z.squeeze(0).squeeze(0)                 # (asset_dim, num_latents, latent_dim)
        a = a.transpose(0, 2)                       # (asset_dim, window_size+1, 1)
        r = r.squeeze(1)                            # (1,)

        s[..., -1] = a.squeeze(-1)[:, :-1]          # Replace the last column with the action
        s_[..., -1] = a.squeeze(-1)[:, 1:]          # Replace the last column with the action

        a = a[:, -1]                                # (asset_dim, 1)

        return s, z, a, r, s_