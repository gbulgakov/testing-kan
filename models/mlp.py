import torch.nn as nn

# обычная MLP
class MLP(nn.Sequential):
    def __init__(self, layers, dropout):
        super(MLP, self).__init__()
        
        total_layers = []
        for n_in, n_out in zip(layers[:-2], layers[1:-1]):
            total_layers.append(nn.Linear(n_in, n_out))
            total_layers.append(nn.SiLU(inplace=False))
            total_layers.append(nn.Dropout(dropout, inplace=False))
        total_layers.append(nn.Linear(layers[-2], layers[-1])) # выходной слой

        self.classifier = nn.Sequential(*total_layers)
