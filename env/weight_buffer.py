import torch

class ActionBuffer:
    def __init__(self, cfg):
        """ Initialize the action buffer """
        self.window = cfg["window_size"]
        self.act_dim = cfg["asset_dim"]

        self.buffer = torch.zeros((self.window, self.act_dim))  # Buffer to store actions
        self.buffer[0, 0] = 1                                   # Init first action as all cash
        self.idx = 1                                            # Pointer to curr position in the buffer
        self.is_full = False                                    # Track if the buffer is full

    def update(self, action):
        """ Update the buffer with the latest action.
        Args:
            action (torch.Tensor): A tensor of shape (act_dim,) representing the action.
        """
        if action.shape != (self.act_dim,):
            raise ValueError(f"Action must have shape ({self.act_dim},), got {action.shape}")
        
        self.buffer[self.idx] = action              # Insert action at the current buffer position
        self.idx = (self.idx + 1) % self.window     # Increment the buffer pointer
        
        # Mark buffer as full after the first full cycle
        if self.idx == 0:
            self.is_full = True

    def get_last(self):
        """ Retrieve the last action from the buffer """
        return self.buffer[(self.idx - 1) % self.window]

    def get_all(self):
        """ Retrieve the buffer in shape (act_dim, window_size) with the most recent actions
            placed at the end of the buffer. Empty slots at beginning are padded with zeros.
        Returns:
            torch.Tensor: Buffer of shape (act_dim, window).
        """
        if self.is_full:
            buffer = self.buffer
        else:
            padding = torch.zeros((self.window - self.idx, self.act_dim))
            buffer = torch.cat((padding, self.buffer[:self.idx]), dim=0)

        return buffer.T

    def reset(self):
        """ Reset the buffer to the initial state """
        self.buffer = torch.zeros((self.window, self.act_dim))
        self.buffer[0, 0] = 1
        self.idx = 1
        self.is_full = False