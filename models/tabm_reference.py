# License: https://github.com/yandex-research/tabm/blob/main/LICENSE

# NOTE
# The minimum required versions of the dependencies are specified in README.md.

import itertools
from typing import Any, Literal
from models.fastkan import FastKAN, FastKANLayer
from models.chebyshev_kan import ChebyKAN, ChebyKANLayer
from models.efficient_kan import KAN, KANLinear
from models.fastkan import SplineLinear
from models.fastkan import RadialBasisFunction
from models.mlp import MLP
import rtdl_num_embeddings
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
from typing import *


# ======================================================================================
# Initialization
# ======================================================================================
def init_rsqrt_uniform_(x: Tensor, d: int) -> Tensor:
    assert d > 0
    d_rsqrt = d**-0.5
    return nn.init.uniform_(x, -d_rsqrt, d_rsqrt)


@torch.inference_mode()
def init_random_signs_(x: Tensor) -> Tensor:
    return x.bernoulli_(0.5).mul_(2).add_(-1)


# ======================================================================================
# Modules
# ======================================================================================
class NLinear(nn.Module):
    """N linear layers applied in parallel to N disjoint parts of the input.

    **Shape**

    - Input: ``(B, N, in_features)``
    - Output: ``(B, N, out_features)``

    The i-th linear layer is applied to the i-th matrix of the shape (B, in_features).

    Technically, this is a simplified version of delu.nn.NLinear:
    https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html.
    The difference is that this layer supports only 3D inputs
    with exactly one batch dimension. By contrast, delu.nn.NLinear supports
    any number of batch dimensions.
    """

    def __init__(
        self, n: int, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        d = self.weight.shape[-2]
        init_rsqrt_uniform_(self.weight, d)
        if self.bias is not None:
            init_rsqrt_uniform_(self.bias, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]

        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


class ScaleEnsemble(nn.Module):
    def __init__(
        self,
        k: int,
        d: int,
        *,
        init: Literal['ones', 'normal', 'random-signs'],
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, d))
        self._weight_init = init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self._weight_init == 'ones':
            nn.init.ones_(self.weight)
        elif self._weight_init == 'normal':
            nn.init.normal_(self.weight)
        elif self._weight_init == 'random-signs':
            init_random_signs_(self.weight)
        else:
            raise ValueError(f'Unknown weight_init: {self._weight_init}')

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 2
        return x * self.weight


class LinearEfficientEnsemble(nn.Module):
    """
    This layer is a more configurable version of the "BatchEnsemble" layer
    from the paper
    "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning"
    (link: https://arxiv.org/abs/2002.06715).

    First, this layer allows to select only some of the "ensembled" parts:
    - the input scaling  (r_i in the BatchEnsemble paper)
    - the output scaling (s_i in the BatchEnsemble paper)
    - the output bias    (not mentioned in the BatchEnsemble paper,
                          but is presented in public implementations)

    Second, the initialization of the scaling weights is configurable
    through the `scaling_init` argument.

    NOTE
    The term "adapter" is used in the TabM paper only to tell the story.
    The original BatchEnsemble paper does NOT use this term. So this class also
    avoids the term "adapter".
    """

    r: None | Tensor
    s: None | Tensor
    bias: None | Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        k: int,
        ensemble_scaling_in: bool,
        ensemble_scaling_out: bool,
        ensemble_bias: bool = True, # WARNING: костыль, чтобы упросить вызов make_efficient_ensemble  далее
        scaling_init: Literal['ones', 'random-signs'],
    ):
        assert k > 0
        if ensemble_bias:
            assert bias
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_parameter(
            'r',
            (
                nn.Parameter(torch.empty(k, in_features))
                if ensemble_scaling_in
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            's',
            (
                nn.Parameter(torch.empty(k, out_features))
                if ensemble_scaling_out
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            'bias',
            (
                nn.Parameter(torch.empty(out_features))  # type: ignore[code]
                if bias and not ensemble_bias
                else nn.Parameter(torch.empty(k, out_features))
                if ensemble_bias
                else None
            ),
        )

        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.scaling_init = scaling_init

        self.reset_parameters()

    def reset_parameters(self):
        init_rsqrt_uniform_(self.weight, self.in_features)
        scaling_init_fn = {'ones': nn.init.ones_, 'random-signs': init_random_signs_}[
            self.scaling_init
        ]
        if self.r is not None:
            scaling_init_fn(self.r)
        if self.s is not None:
            scaling_init_fn(self.s)
        if self.bias is not None:
            bias_init = torch.empty(
                # NOTE: the shape of bias_init is (out_features,) not (k, out_features).
                # It means that all biases have the same initialization.
                # This is similar to having one shared bias plus
                # k zero-initialized non-shared biases.
                self.out_features,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            bias_init = init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, K, D)
        assert x.ndim == 3

        # >>> The equation (5) from the BatchEnsemble paper (arXiv v2).
        if self.r is not None:
            x = x * self.r
        x = x @ self.weight.T
        if self.s is not None:
            x = x * self.s
        # <<<

        if self.bias is not None:
            x = x + self.bias
        return x
    
'''WARNING: Initialization is different from standard ChebyKAN implementation'''
class ChebyKanEnsembleLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int, # number of ensembles   
        degree: int, # degree of chebyshev polynomial
        *,
        ensemble_scaling_in: bool, # whether to scale the input by the number of ensembles  
        ensemble_scaling_out: bool, # whether to scale the output by the number of ensembles
        scaling_init: Literal['ones', 'random-signs'] = 'random-signs', # initialization for scaling
    ):
        assert k > 1
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.k = k
        
        self.cheby_coeffs = nn.Parameter(torch.empty(in_features, out_features, degree + 1)) # coefs for einsum later
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))
        
        self.register_parameter('r', nn.Parameter(torch.empty(k, in_features)) if ensemble_scaling_in else None) # input scaling
        self.register_parameter('s', nn.Parameter(torch.empty(k, out_features)) if ensemble_scaling_out else None) # output scaling
        
        self.scaling_init = scaling_init
        self.reset_parameters()
        
    def reset_parameters(self):
        init_rsqrt_uniform_(self.cheby_coeffs, self.in_features * (self.degree + 1)) #TODO: check different init methods (xavier, random-signs, normal)
        scaling_init_fn = {'ones': nn.init.ones_, 'random-signs': init_random_signs_}[
            self.scaling_init
        ]
        if self.r is not None:
            scaling_init_fn(self.r)
        if self.s is not None:
            scaling_init_fn(self.s)
        
    def forward(self, x: Tensor) -> Tensor:
        #x.shape == (B, k, D)
        assert x.ndim == 3
        
        if self.r is not None:
            x = x * self.r
            
        #chebyshev part
        x = torch.clamp(torch.tanh(x), -1 + 1e-6, 1 - 1e-6)  # безопасный tanh + clamp
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bkid,iod->bko", x, self.cheby_coeffs
        )
        if self.s is not None:
            x = x * self.s
        return x
    
class FastKanEnsembleLayer(nn.Module):
    def __init__(self,
        in_features: int,
        out_features: int,
        k: int, # number of ensembles   
        #parameters for fastkan
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
        *,
        ensemble_scaling_in: bool, # whether to scale the input by the number of ensembles  
        ensemble_scaling_out: bool, # whether to scale the output by the number of ensembles
        scaling_init: Literal['ones', 'random-signs'] = 'random-signs'
    ): # initialization for scaling:
        assert k > 1
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.num_grids = num_grids
        self.grid_max = grid_max
        self.grid_min = grid_min
        self.use_base_update = use_base_update
        self.use_layernorm = use_layernorm
        self.base_activation = base_activation
        self.spline_weight_init_scale = spline_weight_init_scale
        
        self.register_parameter('r', nn.Parameter(torch.empty(k, in_features)) if ensemble_scaling_in else None) # input scaling
        self.register_parameter('s', nn.Parameter(torch.empty(k, out_features)) if ensemble_scaling_out else None) # output scaling
        
        self.scaling_init = scaling_init
        self.reset_parameters()
    
    def reset_parameters(self):
        #from fastkan implementation
        self.layernorm = None
        if self.use_layernorm:
            assert self.in_features > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(self.in_features)
        self.rbf = RadialBasisFunction(self.grid_min, self.grid_max, self.num_grids)
        self.spline_linear = SplineLinear(self.in_features * self.num_grids, self.out_features, self.spline_weight_init_scale)
        self.use_base_update = self.use_base_update
        if self.use_base_update:
            self.base_activation = self.base_activation
            self.base_linear = nn.Linear(self.in_features, self.out_features)
        #from TabM implementation
        scaling_init_fn = {'ones': nn.init.ones_, 'random-signs': init_random_signs_}[
            self.scaling_init
        ]
        if self.r is not None:
            scaling_init_fn(self.r)
        if self.s is not None:
            scaling_init_fn(self.s)
    
    def forward(self, x: Tensor) -> Tensor:
        #x.shape == (B, k, D)
        assert x.ndim == 3
        
        if self.r is not None:
            x = x * self.r
            
        #fastkan part
        if self.layernorm is not None and self.use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        x = ret
        if self.s is not None:
            x = x * self.s
        return x

class EfficientKanEnsembleLayer(nn.Module):
    def __init__(self,
        in_features: int,
        out_features: int,
        k: int, # number of ensembles   
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        *,
        ensemble_scaling_in: bool, # whether to scale the input by the number of ensembles  
        ensemble_scaling_out: bool, # whether to scale the output by the number of ensembles
        scaling_init: Literal['ones', 'random-signs'] = 'random-signs'
    ):
        assert k > 1
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        
        self.register_parameter(
            'r',
            (
                nn.Parameter(torch.empty(k, in_features))
                if ensemble_scaling_in
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            's',
            (
                nn.Parameter(torch.empty(k, out_features))
                if ensemble_scaling_out
                else None
            ),  # type: ignore[code]
        )
        self.reset_parameters()
            
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
        
        scaling_init_fn = {'ones': nn.init.ones_, 'random-signs': init_random_signs_}[
            self.scaling_init
        ]
        if self.r is not None:
            scaling_init_fn(self.r)
        if self.s is not None:
            scaling_init_fn(self.s)
    
    def b_splines(self, x: torch.Tensor):
        original_shape = x.shape  # (batch_size, k, in_features) or (batch_size, in_features)
        x = x.reshape(-1, self.in_features)  # (flattened, in_features)
        
        grid = self.grid  # (in_features, grid_size + 2*spline_order + 1)
        x = x.unsqueeze(-1)
        
        # Вычисление базисов (как в оригинале)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k+1)]) / (grid[:, k:-1] - grid[:, :-(k+1)]) * bases[:, :, :-1]
            ) + (
                (grid[:, k+1:] - x) / (grid[:, k+1:] - grid[:, 1:-k]) * bases[:, :, 1:]
            )
        
        # Восстановление исходной формы + доп измерение
        bases = bases.reshape(*original_shape, -1)
        return bases.contiguous()
    
    
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        print(A.shape, B.shape)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()
    
    
    # def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
    #     """
    #     Compute the coefficients of the curve that interpolates the given points for an ensemble of tasks.

    #     Args:
    #         x (torch.Tensor): Input tensor of shape (batch_size, k, in_features) for ensemble,
    #                         or (batch_size, in_features) for single task.
    #         y (torch.Tensor): Output tensor of shape (batch_size, k, in_features, out_features) for ensemble,
    #                         or (batch_size, in_features, out_features) for single task.

    #     Returns:
    #         torch.Tensor: Coefficients tensor of shape (k, out_features, in_features, grid_size + spline_order) for ensemble,
    #                     or (out_features, in_features, grid_size + spline_order) for single task.
    #     """
    #     # Handle single task case (add k=1 dimension)
    #     if x.dim() == 2:
    #         x = x.unsqueeze(1)  # (batch_size, 1, in_features)
    #     if y.dim() == 3:
    #         y = y.unsqueeze(1)  # (batch_size, 1, in_features, out_features)
        
    #     batch_size, k, in_features = x.shape
    #     _, _, _, out_features = y.shape
        
    #     assert x.size(0) == y.size(0) and x.size(1) == y.size(1)
    #     assert x.size(2) == self.in_features
    #     assert y.size(2) == self.in_features
    #     assert y.size(3) == self.out_features

    #     # Compute B-splines for all x in the ensemble
    #     # A shape: (batch_size, k, in_features, grid_size + spline_order)
    #     A = self.b_splines(x.flatten(0, 1)).view(
    #         batch_size, k, in_features, -1
    #     )
        
    #     # Prepare tensors for solving
    #     A = A.permute(2, 0, 1, 3)  # (in_features, batch_size, k, grid_size + spline_order)
    #     B = y.permute(2, 0, 1, 3)  # (in_features, batch_size, k, out_features)
        

    #     # Solve all systems in parallel
    #     result = torch.linalg.lstsq(
    #         A, B
    #     ).solution  # (out_features, in_features, k, grid_size + spline_order)
        
    #     return result.contiguous()
    
    def forward(self, x: torch.Tensor):
        #x.shape == (B, k, D)
        assert x.ndim == 3
        
        if self.r is not None:
            x = x * self.r
            
        #efficientkan part
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        x = output.reshape(*original_shape[:-1], self.out_features)
        
        if self.s is not None:
            x = x * self.s
        return x



def make_efficient_ensemble(module: nn.Module, EnsembleLayer, **kwargs) -> None:
    """Replace linear layers with efficient ensembles of linear layers.

    NOTE
    In the paper, there are no experiments with networks with normalization layers.
    Perhaps, their trainable weights (the affine transformations) also need
    "ensemblification" as in the paper about "FiLM-Ensemble".
    Additional experiments are required to make conclusions.
    """
    for name, submodule in list(module.named_children()):
        if isinstance(submodule, nn.Linear) or isinstance(submodule, ChebyKANLayer) or isinstance(submodule, FastKANLayer) or isinstance(submodule, KANLinear):
            module.add_module(
                name,
                EnsembleLayer(
                    in_features=submodule.in_features,
                    out_features=submodule.out_features,
                    # bias=submodule.bias is not None,
                    **kwargs,
                ),
            )
        else:
            make_efficient_ensemble(submodule, EnsembleLayer, **kwargs)


def _get_first_ensemble_layer(backbone: MLP) -> LinearEfficientEnsemble:
    if isinstance(backbone, MLP):
        return backbone.blocks[0][0]  # type: ignore[code]
    else:
        raise RuntimeError(f'Unsupported backbone: {backbone}')


@torch.inference_mode()
def _init_first_adapter(
    weight: Tensor,
    distribution: Literal['normal', 'random-signs'],
    init_sections: list[int],
) -> None:
    """Initialize the first adapter.

    NOTE
    The `init_sections` argument is a historical artifact that accidentally leaked
    from irrelevant experiments to the final models. Perhaps, the code related
    to `init_sections` can be simply removed, but this was not tested.
    """
    assert weight.ndim == 2
    assert weight.shape[1] == sum(init_sections)

    if distribution == 'normal':
        init_fn_ = nn.init.normal_
    elif distribution == 'random-signs':
        init_fn_ = init_random_signs_
    else:
        raise ValueError(f'Unknown distribution: {distribution}')

    section_bounds = [0, *torch.tensor(init_sections).cumsum(0).tolist()]
    for i in range(len(init_sections)):
        # NOTE
        # As noted above, this section-based initialization is an arbitrary historical
        # artifact. Consider the first adapter of one ensemble member.
        # This adapter vector is implicitly split into "sections",
        # where one section corresponds to one feature. The code below ensures that
        # the adapter weights in one section are initialized with the same random value
        # from the given distribution.
        w = torch.empty((len(weight), 1), dtype=weight.dtype, device=weight.device)
        init_fn_(w)
        weight[:, section_bounds[i] : section_bounds[i + 1]] = w


_CUSTOM_MODULES = {
    # https://docs.python.org/3/library/stdtypes.html#definition.__name__
    CustomModule.__name__: CustomModule
    for CustomModule in [
        rtdl_num_embeddings.LinearEmbeddings,
        rtdl_num_embeddings.LinearReLUEmbeddings,
        rtdl_num_embeddings.PeriodicEmbeddings,
        rtdl_num_embeddings.PiecewiseLinearEmbeddings,
        MLP,
        KAN,
        FastKAN,
        ChebyKAN,
    ]
}


def make_module(type: str, *args, **kwargs) -> nn.Module:
    Module = getattr(nn, type, None)
    if Module is None:
        Module = _CUSTOM_MODULES[type]
    return Module(*args, **kwargs)


# ======================================================================================
# Optimization
# ======================================================================================
def default_zero_weight_decay_condition( # выбирает слои, которые не будут иметь weight_decay
    module_name: str, module: nn.Module, parameter_name: str, parameter: nn.Parameter
):
    from rtdl_num_embeddings import _Periodic

    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        nn.BatchNorm1d
        | nn.LayerNorm
        | nn.InstanceNorm1d
        | rtdl_num_embeddings.LinearEmbeddings
        | rtdl_num_embeddings.LinearReLUEmbeddings
        | _Periodic,
    )


def make_parameter_groups(
    module: nn.Module,
    zero_weight_decay_condition=default_zero_weight_decay_condition,
    custom_groups: None | list[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    Creates parameter groups for the optimizer.
    Parameters that satisfy zero_weight_decay_condition will have weight_decay=0.
    """
    if custom_groups is None:
        custom_groups = []
    custom_params = frozenset(
        itertools.chain.from_iterable(group['params'] for group in custom_groups)
    )
    assert len(custom_params) == sum(
        len(group['params']) for group in custom_groups
    ), 'Parameters in custom_groups must not intersect'
    zero_wd_params = frozenset(
        p
        for mn, m in module.named_modules()
        for pn, p in m.named_parameters()
        if p not in custom_params and zero_weight_decay_condition(mn, m, pn, p)
    )
    default_group = {
        'params': [
            p
            for p in module.parameters()
            if p not in custom_params and p not in zero_wd_params
        ]
    }
    return [
        default_group,
        {'params': list(zero_wd_params), 'weight_decay': 0.0},
        *custom_groups,
    ]


# ======================================================================================
# The model
# ======================================================================================
class Model(nn.Module):
    """MLP & TabM."""

    def __init__(
        self,
        *,
        n_num_features: int,
        # n_classes: None | int, already in backbone
        backbone: nn.Module,
        bins: None | list[Tensor],  # For piecewise-linear encoding/embeddings.
        num_embeddings: None | dict = None,
        arch_type: Literal[
            # Plain feed-forward network without any kind of ensembling.
            'plain',
            #
            # TabM
            'tabm',
            #
            # TabM-mini
            'tabm-mini',
            #
            # TabM-packed
            'tabm-packed',
            #
            # TabM. The first adapter is initialized from the normal distribution.
            # This variant was not used in the paper, but it may be useful in practice.
            'tabm-normal',
            #
            # TabM-mini. The adapter is initialized from the normal distribution.
            # This variant was not used in the paper.
            'tabm-mini-normal',
        ],
        k: None | int = None,
        share_training_batches: bool = True,
        **kwargs # доп параметры для chebykan, fastkan, efficientkan
    ) -> None:
        # >>> Validate arguments.
        assert n_num_features >= 0
        if arch_type == 'plain':
            assert k is None
            assert (
                share_training_batches
            ), 'If `arch_type` is set to "plain", then `simple` must remain True'
        else:
            assert k is not None
            assert k > 0

        super().__init__()

        # >>> Continuous (numerical) features
        first_adapter_sections = []  # See the comment in `_init_first_adapter`.

        if n_num_features == 0:
            assert bins is None
            self.num_module = None
            d_num = 0

        elif num_embeddings is None:
            assert bins is None
            self.num_module = None
            d_num = n_num_features
            first_adapter_sections.extend(1 for _ in range(n_num_features))

        else:
            if bins is None:
                self.num_module = make_module(
                    **num_embeddings
                )
            else:
                assert num_embeddings['type'].startswith('PiecewiseLinearEmbeddings')
                self.num_module = make_module(**num_embeddings, bins=bins)
            d_num = n_num_features * num_embeddings['d_embedding']
            first_adapter_sections.extend(
                num_embeddings['d_embedding'] for _ in range(n_num_features)
            )

        '''Уже обработаны категориальные признаки, так что делаем вид, что их нет'''
        # # >>> Categorical features
        # self.cat_module = (
        #     OneHotEncoding0d(cat_cardinalities) if cat_cardinalities else None
            
        # )
        # first_adapter_sections.extend(cat_cardinalities)
        # d_cat = sum(cat_cardinalities)

        # >>> Backbone
        self.minimal_ensemble_adapter = None
        '''Передаем модель из вне! (Как раньше)'''
        self.backbone = backbone

        if arch_type != 'plain':
            assert k is not None
            first_adapter_init = (
                None
                if arch_type == 'tabm-packed'
                else 'normal'
                if arch_type in ('tabm-mini-normal', 'tabm-normal')
                # For other arch_types, the initialization depends
                # on the presense of num_embeddings.
                else 'random-signs'
                if num_embeddings is None
                else 'normal'
            )

            if arch_type in ('tabm', 'tabm-normal'):
                # Like BatchEnsemble, but all multiplicative adapters,
                # except for the very first one, are initialized with ones.
                assert first_adapter_init is not None
                # подбираем тип эффективного слоя в зависимости от типа модели
                efficient_layer = (
                    LinearEfficientEnsemble
                    if isinstance(self.backbone, MLP)
                    else ChebyKanEnsembleLayer
                    if isinstance(self.backbone, ChebyKAN)
                    else FastKanEnsembleLayer
                    if isinstance(self.backbone, FastKAN)
                    else EfficientKanEnsembleLayer
                )
                make_efficient_ensemble(
                    self.backbone,
                    efficient_layer,
                    k=k,
                    ensemble_scaling_in=True,
                    ensemble_scaling_out=True,
                    # ensemble_bias=True, пофиксили передачей дефолтного значение в линейную модель
                    scaling_init='ones',
                    **kwargs
                )
                _init_first_adapter(
                    _get_first_ensemble_layer(self.backbone).r,  # type: ignore[code]
                    first_adapter_init,
                    first_adapter_sections,
                )

            elif arch_type in ('tabm-mini', 'tabm-mini-normal'):
                # MiniEnsemble
                assert first_adapter_init is not None
                self.minimal_ensemble_adapter = ScaleEnsemble(
                    k,
                    d_flat,
                    init='random-signs' if num_embeddings is None else 'normal',
                )
                _init_first_adapter(
                    self.minimal_ensemble_adapter.weight,  # type: ignore[code]
                    first_adapter_init,
                    first_adapter_sections
                )
            elif arch_type == 'tabm-packed': # future work
                  raise ValueError(f'tabm-packed is not supported yet')
            
            #     # Packed ensemble.
            #     # In terms of the Packed Ensembles paper by Laurent et al.,
            #     # TabM-packed is PackedEnsemble(alpha=k, M=k, gamma=1).
            #     assert first_adapter_init is None
            #     make_efficient_ensemble(self.backbone, NLinear, n=k, **kwargs)
            else:
                raise ValueError(f'Unknown arch_type: {arch_type}')

        '''Backbone is a complete model with correct output layer'''
        # >>> Output
        # d_block = backbone['d_block']
        # d_out = 1 if n_classes is None else n_classes
        # self.output = (
        #     nn.Linear(d_block, d_out)
        #     if arch_type == 'plain'
        #     else NLinear(k, d_block, d_out)  # type: ignore[code]
        # )

        # >>>
        self.arch_type = arch_type
        self.k = k
        self.share_training_batches = share_training_batches

    def forward(
        self, x_num: None | Tensor = None, x_cat: None | Tensor = None
    ) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num if self.num_module is None else self.num_module(x_num))
        if x_cat is not None:
            x.append((x_cat).float())
        x = torch.column_stack([x_.flatten(1, -1) for x_ in x])

        if self.k is not None:
            if self.share_training_batches or not self.training:
                # (B, D) -> (B, K, D)
                x = x[:, None].expand(-1, self.k, -1)
            else:
                # (B * K, D) -> (B, K, D)
                x = x.reshape(len(x) // self.k, self.k, *x.shape[1:])
            if self.minimal_ensemble_adapter is not None:
                x = self.minimal_ensemble_adapter(x)
        else:
            assert self.minimal_ensemble_adapter is None

        x = self.backbone(x)
        # x = self.output(x)
        if self.k is None:
            # Adjust the output shape for plain networks to make them compatible
            # with the rest of the script (loss, metrics, predictions, ...).
            # (B, D_OUT) -> (B, 1, D_OUT)
            x = x[:, None]
        return x
