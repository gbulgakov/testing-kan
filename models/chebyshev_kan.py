import torch
import torch.nn as nn


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree):
        super(ChebyKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(in_features, out_features, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (in_features * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.clamp(torch.tanh(x), -1 + 1e-6, 1 - 1e-6)  # безопасный tanh + clamp
        # View and repeat input degree + 1 times
        x = x.view((-1, self.in_features, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, in_features, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, out_features)
        y = y.view(-1, self.out_features)
        return y

class ChebyKAN(nn.Module):
    def __init__(self, layers_hidden, degree):
        super(ChebyKAN, self).__init__()
        self.degree = degree

        self.layers = nn.ModuleList([
            ChebyKANLayer(in_features, out_features, degree) 
            for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
