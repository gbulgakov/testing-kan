from __future__ import annotations

import gc
import os
import random
import zipfile
from typing import Dict, Any, Iterable

import numpy as np
import torch

# Optional: custom optimizers
from .optimizers.ademamix import AdEMAMix
from .optimizers.muon import Muon
from .optimizers.mars import MARS

# --------------------------------------------------------------------------- #
# Optimizer registry
# --------------------------------------------------------------------------- #

OPTIMIZERS: Dict[str, Any] = {
    "adamw": torch.optim.AdamW,
    "ademamix": AdEMAMix,
    "mars": MARS,
    "muon": Muon,
}

# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #


def create_zip_archive(source_dir: str | os.PathLike, archive_path: str | os.PathLike) -> None:
    """
    Collect all files inside `source_dir` into a single ZIP archive.
    """
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for fname in os.listdir(source_dir):
            fpath = os.path.join(source_dir, fname)
            if os.path.isfile(fpath) and not fname.endswith(".zip"):
                # `arcname=fname` keeps the root of the archive clean
                zipf.write(fpath, arcname=fname)


# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #


def compare_epochs(task_type: str, epoch_a: Dict[str, Any], epoch_b: Dict[str, Any]) -> bool:
    """
    Return *True* if `epoch_a` outperforms `epoch_b`.
    """
    if task_type == "regression":
        return epoch_a["loss"] < epoch_b["loss"]
    return epoch_a["acc"] > epoch_b["acc"]


def seed_everything(seed: int = 0) -> None:
    """
    Set seed for Python / NumPy / PyTorch to get reproducible results.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(optim_name: str, model_params: Iterable[torch.nn.Parameter], config: Dict[str, Any]):
    """
    Instantiate an optimizer from `OPTIMIZERS`.

    Expected keys in `config`
    -------------------------
    lr : float
    weight_decay : float   (ignored for Muon)
    """
    optim_class = OPTIMIZERS[optim_name]
    optim_kwargs: Dict[str, Any] = {"lr": config["lr"]}

    # Weight decay is not part of Muon’s API
    if optim_name != "muon":
        optim_kwargs["weight_decay"] = config["weight_decay"]

    # Muon expects a list, not a generator
    if optim_name == "muon":
        model_params = list(model_params)

    return optim_class(model_params, **optim_kwargs)
    
 
def clean_up_model(model: torch.nn.Module, optimizer) -> None:
    """
    Free GPU memory after a training run.
    """
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()
