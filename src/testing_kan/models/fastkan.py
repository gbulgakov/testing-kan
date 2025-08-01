# Copyright 2024 Li, Ziyao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *

class SplineLinear(nn.Linear):
    def __init__(self, input_dim: int, output_dim: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(input_dim, output_dim, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim # needed for make_efficient_ensemble
        self.output_dim = output_dim # needed for make_efficient_ensemble
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2
    ):
        '''this function returns the learned curves in a FastKANLayer.
        input_index: the selected index of the input, in [0, input_dim) .
        output_index: the selected index of the output, in [0, output_dim) .
        num_pts: num of points sampled for the curve.
        num_extrapolate_bins (N_e): num of bins extrapolating from the given grids. The curve 
            will be calculate in the range of [grid_min - h * N_e, grid_max + h * N_e].
        '''
        ng = self.rbf.num_grids
        h = self.rbf.denominator
        assert input_index < self.input_dim
        assert output_index < self.output_dim
        w = self.spline_linear.weight[
            output_index, input_index * ng : (input_index + 1) * ng
        ]   # num_grids,
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts
        )   # num_pts, num_grids
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)
        return x, y
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min: float, grid_max: float, num_grids: int):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        h = (grid_max - grid_min) / (num_grids - 1) if num_grids > 1 else 0.0
        self.register_buffer("h", torch.tensor(h))
        self.register_buffer("grid", torch.linspace(grid_min, grid_max, num_grids))

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1)  # [..., num_features, 1]
        distances = (x - self.grid) / self.h  # [..., num_features, num_grids]
        return torch.exp(-(distances ** 2))  # Gaussian RBF

class _NFastKANLayer(nn.Module):
    def __init__(
        self,
        n: int,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = False,
        base_activation=F.silu,
        spline_weight_init_scale: float = 0.1,
    ):
        super().__init__()
        self.n = n
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_grids = num_grids
        self.use_base_update = use_base_update
        self.base_activation = base_activation

        # RBF initialization
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        
        # LayerNorm
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "LayerNorm requires input_dim > 1"
            self.layernorm = nn.LayerNorm(input_dim)

        # Spline transition params
        self.spline_weight = nn.Parameter(
            torch.empty(n, output_dim, input_dim * num_grids)
        )
        nn.init.normal_(self.spline_weight, std=spline_weight_init_scale)

        # Base transition params
        if self.use_base_update:
            self.base_weight = nn.Parameter(torch.empty(n, output_dim, input_dim))
            self.base_bias = nn.Parameter(torch.empty(n, output_dim))
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.base_bias, -bound, bound)
        else:
            self.register_parameter("base_weight", None)
            self.register_parameter("base_bias", None)

    def forward(self, x: torch.Tensor, use_layernorm: bool = True) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        # x: [batch_size, n, input_dim]
        original_shape = x.shape
        x = x.view(-1, self.n, self.input_dim)

        #  LayerNorm
        if self.layernorm is not None and use_layernorm:
            x = self.layernorm(x)

        # Spline transition
        rbf_out = self.rbf(x)  # [batch_size, n, input_dim, num_grids]
        rbf_flat = rbf_out.view(*rbf_out.shape[:-2], -1)  # [batch_size, n, input_dim*num_grids]
        spline_out = torch.einsum("bni,noi->bno", rbf_flat, self.spline_weight)

        # Base transition
        if self.use_base_update:
            base = torch.einsum(
                "bni,noi->bno",
                self.base_activation(x),
                self.base_weight,
            )
            spline_out = spline_out + base + self.base_bias.unsqueeze(0)

        return spline_out.view(*original_shape[:-1], self.output_dim)


class FastKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AttentionWithFastKANTransform(nn.Module):
    
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = True,
    ):
        super(AttentionWithFastKANTransform, self).__init__()

        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.linear_q = FastKANLayer(q_dim, total_dim)
        self.linear_k = FastKANLayer(k_dim, total_dim)
        self.linear_v = FastKANLayer(v_dim, total_dim)
        self.linear_o = FastKANLayer(total_dim, q_dim)
        self.linear_g = None
        if self.gating:
            self.linear_g = FastKANLayer(q_dim, total_dim)
        # precompute the 1/sqrt(head_dim)
        self.norm = head_dim**-0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor = None,      # additive attention bias
    ) -> torch.Tensor:         

        wq = self.linear_q(q).view(*q.shape[:-1], 1, self.num_heads, -1) * self.norm     # *q1hc
        wk = self.linear_k(k).view(*k.shape[:-2], 1, k.shape[-2], self.num_heads, -1)    # *1khc
        att = (wq * wk).sum(-1).softmax(-2)     # *qkh
        del wq, wk
        if bias is not None:
            att = att + bias[..., None]

        wv = self.linear_v(v).view(*v.shape[:-2],1, v.shape[-2], self.num_heads, -1)     # *1khc
        o = (att[..., None] * wv).sum(-3)        # *qhc
        del att, wv

        o = o.view(*o.shape[:-2], -1)           # *q(hc)

        if self.linear_g is not None:
            # gating, use raw query input
            g = self.linear_g(q)
            o = torch.sigmoid(g) * o

        # merge heads
        o = self.linear_o(o)
        return o
