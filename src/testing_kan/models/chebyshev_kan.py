from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ChebyKANLayer(nn.Module):
    '''
    Single ChebyKAN Layer
    '''
    def __init__(self, in_features: int, out_features: int, degree: int) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        # Learnable Chebyshev coefficients: (I, O, D)
        self.cheby_coeffs = nn.Parameter(
            torch.empty(in_features, out_features, degree + 1)
        )
        nn.init.normal_(
            self.cheby_coeffs, mean=0.0, std=1 / (in_features * (degree + 1))
        )

        # Pre-compute range [0, 1, …, degree] for broadcasting during forward
        self.register_buffer("arange", torch.arange(0, degree + 1))

    def forward(self, x: Tensor) -> Tensor:  # (B, I) → (B, O)
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        # HERE we added constant 1e-6 to ensure numerical stability!!
        x = torch.clamp(torch.tanh(x), -1 + 1e-6, 1 - 1e-6)

        # Repeat along the polynomial dimension: (B, I, D)
        x = x.view(-1, self.in_features, 1).expand(-1, -1, self.degree + 1)

        # T_k(x) = cos(k * arccos(x))
        x = (x.acos() * self.arange).cos()  # still (B, I, D)

        # Linear combination with learnable coefficients
        # einsum: (B I D, I O D) → (B O)
        y = torch.einsum("bid,iod->bo", x, self.cheby_coeffs)

        return y
    

class ChebyKAN(nn.Module):
    """
    Simple multi-layer Chebyshev-KAN network.
    """

    def __init__(self, layers_hidden: list[int], degree: int) -> None:
        super().__init__()
        self.degree = degree

        self.layers = nn.ModuleList(
            [
                ChebyKANLayer(inp, out, degree)
                for inp, out in zip(layers_hidden[:-1], layers_hidden[1:])
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
