import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
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

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    # def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
    #     """
    #     Compute the coefficients of the curve that interpolates the given points.

    #     Args:
    #         x (torch.Tensor): Input tensor of shape (batch_size, in_features).
    #         y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

    #     Returns:
    #         torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
    #     """
    #     assert x.dim() == 2 and x.size(1) == self.in_features
    #     assert y.size() == (x.size(0), self.in_features, self.out_features)

    #     A = self.b_splines(x).transpose(
    #         0, 1
    #     )  # (in_features, batch_size, grid_size + spline_order)
    #     B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
    #     solution = torch.linalg.lstsq(
    #         A, B
    #     ).solution  # (in_features, grid_size + spline_order, out_features)
    #     result = solution.permute(
    #         2, 0, 1
    #     )  # (out_features, in_features, grid_size + spline_order)

    #     assert result.size() == (
    #         self.out_features,
    #         self.in_features,
    #         self.grid_size + self.spline_order,
    #     )
    #     return result.contiguous()
    
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points for an ensemble of tasks.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, k, in_features) for ensemble,
                            or (batch_size, in_features) for single task.
            y (torch.Tensor): Output tensor of shape (batch_size, k, in_features, out_features) for ensemble,
                            or (batch_size, in_features, out_features) for single task.

        Returns:
            torch.Tensor: Coefficients tensor of shape (k, out_features, in_features, grid_size + spline_order) for ensemble,
                        or (out_features, in_features, grid_size + spline_order) for single task.
        """
        # Handle single task case (add k=1 dimension)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, in_features)
        if y.dim() == 3:
            y = y.unsqueeze(1)  # (batch_size, 1, in_features, out_features)
        
        batch_size, k, in_features = x.shape
        _, _, _, out_features = y.shape
        
        assert x.size(0) == y.size(0) and x.size(1) == y.size(1)
        assert x.size(2) == self.in_features
        assert y.size(2) == self.in_features
        assert y.size(3) == self.out_features

        # Compute B-splines for all x in the ensemble
        # A shape: (batch_size, k, in_features, grid_size + spline_order)
        A = self.b_splines(x.flatten(0, 1)).view(
            batch_size, k, in_features, -1
        )
        
        # Prepare tensors for batch solving
        print(A.shape, B.shape)
        A = A.permute(2, 0, 1)  # (in_features, batch_size, k, grid_size + spline_order)
        B = y.transpose(2, 0, 1)  # (in_features, batch_size, k, out_features)
        
        # Reshape for batch solving: combine batch and in_features dimensions
        # A_flat = A.reshape(-1, k, self.grid_size + self.spline_order)
        # B_flat = B.reshape(-1, k, out_features)
        
        # Solve all systems in parallel
        result = torch.linalg.lstsq(
            A, B
        ).solution  # (out_features, in_features, k, grid_size + spline_order)
        
        # Reshape back and permute dimensions
        # solution = solution.view(
        #     batch_size, in_features, self.grid_size + self.spline_order, out_features
        # )
        # result = solution.permute(3, 1, 2, 0)  # (out_features, in_features, grid_size + spline_order, batch_size)
        
        # # For ensemble, we need to average over batch dimension or handle differently
        # # Here I assume we want separate coefficients for each ensemble member
        # # So we transpose to get (k, out_features, in_features, grid_size + spline_order)
        # result = result.permute(3, 0, 1, 2)  # (batch_size, out_features, in_features, grid_size + spline_order)
        
        # # If we want to average over batch (common approach):
        # result = result.mean(dim=0)  # (out_features, in_features, grid_size + spline_order)
        
        # # Or if we want to keep ensemble dimension:
        # # result = result.permute(1, 2, 3, 0)  # (out_features, in_features, grid_size + spline_order, batch_size)
        # # Then you might need to decide how to combine these
        
        # assert result.size() == (
        #     self.out_features,
        #     self.in_features,
        #     self.grid_size + self.spline_order,
        # )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        batch_norm=True,
        update_grid=False
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.update_grid = update_grid
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(in_features))
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            if self.update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
