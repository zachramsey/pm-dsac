
import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()

        input_dim = cfg["input_dim"]
        hidden_dim = cfg["hidden_dim"]
        output_dim = cfg["output_dim"]
        num_layers = cfg["num_layers"]
        activation = getattr(nn, cfg["activation"])
        output_activation = getattr(nn, cfg["output_activation"])

        self.layers = nn.Sequential()
        self.layers.add_module("input", nn.Linear(input_dim, hidden_dim))
        self.layers.add_module("input_activation", activation())
        for i in range(num_layers - 1):
            self.layers.add_module(f"hidden_{i}", nn.Linear(hidden_dim, hidden_dim))
            self.layers.add_module(f"hidden_{i}_activation", activation())
        self.layers.add_module("output", nn.Linear(hidden_dim, output_dim))
        self.layers.add_module("output_activation", output_activation())

    def forward(self, x):
        return self.layers(x)