from torch import nn
from models.misc import _get_clones


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.activations = _get_clones(activation, num_layers - 1)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activations[i](layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
