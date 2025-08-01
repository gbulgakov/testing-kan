import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layers, dropout):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_in, n_out),
                    nn.SiLU(inplace=False),
                    nn.Dropout(dropout, inplace=False)
                )
                for n_in, n_out in zip(layers[:-2], layers[1:-1])
            ]
        )
        self.layers.append(nn.Linear(layers[-2], layers[-1]))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
