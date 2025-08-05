from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import rtdl_num_embeddings

from .efficient_kan import KAN
from .fastkan import FastKAN
from .mlp import MLP
from .chebyshev_kan import ChebyKAN

# --------------------------------------------------------------------------- #
# Model builder
# --------------------------------------------------------------------------- #


def model_init_preparation(
    config: Dict[str, Any],
    dataset: Dict[str, Any],
    model_name: str,
    arch_type: str,
    emb_name: str,
) -> Tuple[
    Optional[List[int]],                     # layer_widths
    Dict[str, Any],                         # layer_kwargs
    nn.Module,                              # backbone
    Optional[Tensor],                       # bins  (for PLE embeddings)
    Optional[Dict[str, Any]],               # num_embeddings config
    Callable[[Tensor, Tensor], Tensor],     # loss_fn 
    Optional[int],                          # k (ensemble size if arch_type != plain)
]:
    """
    Build backbone network, numerical embeddings and loss-function
    configuration based on a unified *config* dictionary.

    Parameters
    ----------
    config
        Hyper-parameters chosen by the tuner / provided manually.
    dataset
        Pre-processed dataset dict produced by ``data_processing.load_dataset``.
    model_name
        One of ``{"kan", "small_kan", "batch_norm_kan", "fast_kan",
        "cheby_kan", "mlp", "mlp_kan", "kan_mlp"}``.
    arch_type
        ``"plain"`` (single network) or any ensemble variant.
    emb_name
        Numerical embedding type. ``"none"`` means raw features.

    Returns
    -------
    layer_widths
        Complete list of hidden sizes fed back to the caller (for logging).
    layer_kwargs
        Keyword arguments used when instantiating the backbone.
    backbone
        Ready-to-train ``nn.Module``.
    bins
        Pre-computed bin borders for piecewise embeddings (or *None*).
    num_embeddings
        Dict describing the embedding layer for ``rtdl`` (or *None*).
    loss_fn
        Criterion function from ``torch.nn.functional``.
    k 
        Ensemble size (``None`` if *plain* architecture).
    """
    # ------------------------------------------------------------------ dataset
    info = dataset["info"]
    num_cont_cols: int = info["num_cont_cols"]
    num_cat_cols: int = info["num_cat_cols"]

    out_features = 1
    if info["task_type"] == "multiclass":
        out_features = info["n_classes"]
    
    in_features = num_cont_cols * config.get("d_embedding", 1) + num_cat_cols

    layer_widths: Optional[List[int]] = None
    layer_kwargs: Dict[str, Any] = {}
    backbone: nn.Module

    # ------------------------------------------------------------------ KANs / MLP
 
    if model_name == 'kan' or model_name == 'small_kan':
        layer_widths = [in_features] + [config["kan_width"]] * config["kan_layers"] + [out_features]
        layer_kwargs = {'grid_size' : config['grid_size']}
        backbone = KAN(layer_widths, batch_norm=False, **layer_kwargs)
    

    elif model_name == 'fast_kan':
        layer_widths = [in_features] + [config["kan_width"]] * config["kan_layers"] + [out_features]
        layer_kwargs = {'num_grids' : config['grid_size']}
        backbone = FastKAN(layer_widths, **layer_kwargs)

    elif model_name == "cheby_kan":
        layer_widths = [in_features] + [config["kan_width"]] * config["kan_layers"] + [out_features]
        layer_kwargs = {"degree": config["degree"]}
        backbone = ChebyKAN(layers_hidden=layer_widths, **layer_kwargs)
    
    elif model_name == "mlp":
        layer_widths = [in_features] + [config["mlp_width"]] * config["mlp_layers"] + [out_features]
        dropout = config["dropout"]
        backbone = MLP(layer_widths, dropout)
    
    else:
        raise ValueError(f"Unknown model_name '{model_name}'")

    # ------------------------------------------------------------------ embeddings
    X_num = dataset["train"]["X_num"]
    Y = dataset["train"]["y"]
    bins: Optional[Tensor] = None
    num_embeddings: Optional[Dict[str, Any]] = None

    if emb_name in {"piecewiselinearq", "PLE-Q"}:
        bins = rtdl_num_embeddings.compute_bins(X=X_num, n_bins=config["d_embedding"])
        num_embeddings = {
            "type": "PiecewiseLinearEmbeddings",
            "d_embedding": config["d_embedding"],
            "activation": False,
            "version": "B",
        }

    elif emb_name == "periodic":  # PLR
        num_embeddings = {
            "type": "PeriodicEmbeddings",
            "d_embedding": config["d_embedding"],
            "lite": True,
            "frequency_init_scale": config["sigma"],
            "n_features": num_cont_cols,
        }

    elif emb_name == "kan_emb":
        num_embeddings = {
            "type": "_NKANLinear",
            "in_features": 1,
            "out_features": config["d_embedding"],
            "grid_size": config["emb_grid_size"],
            "n": num_cont_cols,
        }

    elif emb_name == "fast_kan_emb":
        num_embeddings = {
            "type": "_NFastKANLayer",
            "input_dim": 1,
            "output_dim": config["d_embedding"],
            "num_grids": config["emb_grid_size"],
            "n": num_cont_cols,
        }

    # ------------------------------------------------------------------ loss
    task_type = info["task_type"]
    if task_type == 'binclass':
        loss_fn = F.binary_cross_entropy_with_logits
    elif task_type == 'multiclass':
        loss_fn = F.cross_entropy
    else:
        loss_fn =  F.mse_loss

    # Ensemble size (only for non-plain architectures)
    k: Optional[int] = 16 if arch_type != "plain" else None
        
    return layer_widths, layer_kwargs, backbone, bins, num_embeddings, loss_fn, k
    
