from typing import Dict, Any, List
import numpy as np
import torch

from src.testing_kan.utils import (
    get_optimizer,
    seed_everything,
    clean_up_model,
)
from .train import train
from src.testing_kan.data_processing import get_dataloaders
from src.testing_kan.models.prepare_model import model_init_preparation
from src.testing_kan.models.tabm_reference import Model


def collect_hyperparams(
    best_params: Dict[str, Any],
    model_name: str,
    emb_name: str,
) -> Dict[str, Any]:
    """
    Convert Optuna `best_params` into the compact hyperparameter
    dictionary expected by model builders and loggers.
    """
    if model_name == 'mlp':
        # MLP-specific keys
        hyperparams = {
            'width' : best_params.get('mlp_width'),
            'hidden_layers' : best_params.get('mlp_layers'),
            'use_dropout' : best_params.get('use_dropout'),
            'dropout' : (0 if not best_params.get('use_dropout') else best_params.get('dropout'))
        }
    else:
        # KAN family
        hyperparams = {
            'width' : best_params.get('kan_width'),
            'hidden_layers' : best_params.get('kan_layers'),
            'grid_size' : best_params.get('grid_size')
        }

    # Embedding-related parameters
    if emb_name != "none":
        hyperparams["d_embedding"] = best_params["d_embedding"]
        if emb_name == "periodic":
            hyperparams["sigma"] = best_params["sigma"]
        if emb_name in {"kan_emb", "fast_kan_emb"}:
            hyperparams["emb_grid_size"] = best_params["emb_grid_size"]

    # Optimizer settings
    hyperparams.update(
        {
            "lr": best_params["lr"],
            "weight_decay": best_params["weight_decay"],
        }
    )
    return hyperparams


def test_best_model(
    *,
    best_params: Dict[str, Any],
    model_name: str,
    arch_type: str,
    emb_name: str,
    optim_name: str,
    dataset: Dict[str, Any],
    num_epochs: int = 100,
    patience: int = 5,
) -> Dict[str, Any]:
    """
    Re-train the best configuration with 10 different seeds and
    report aggregated statistics.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task_type = dataset['info']['task_type']

    num_cont_cols = dataset['info']['num_cont_cols']
    num_cat_cols = dataset['info']['num_cat_cols']

    # preparing logging
    if task_type != 'regression':
        test_best_accuracies: List[float] = []
    test_best_losses: List[float] = []
    val_epoch_times: List[float] = []
    train_epoch_times: List[float] = []
    train_full_times: List[float] = []
    train_epochs: List[int] = []
    best_test_epochs: List[int] = []

    num_params = 0  # will be overwritten inside the loop

    for seed in range(10):
        # Determinism
        seed_everything(seed)

        (
            _,
            layer_kwargs,
            backbone,
            bins,
            embeddings_kwargs,
            loss_fn,
            k,
        ) = model_init_preparation(
            config=best_params,
            dataset=dataset,
            model_name=model_name,
            arch_type=arch_type,
            emb_name=emb_name,
        )

        model = Model(
            n_num_features=num_cont_cols,
            n_cat_features=num_cat_cols,
            backbone=backbone,
            bins=bins,
            num_embeddings=embeddings_kwargs,
            arch_type=arch_type,
            k=k,
            **layer_kwargs,
        )

        optimizer = get_optimizer(optim_name, model.parameters(), best_params)
        real_dataset = get_dataloaders(
            dataset=dataset,
            model=model,
            device=device,
            num_workers=4)

        # Dataloaders are rebuilt for every seed to incorporate any
        # stochastic preprocessing steps.
        real_dataset = get_dataloaders(
            dataset=dataset, model=model, device=device, num_workers=4
        )
        real_dataset["info"] = dataset["info"]

        # Train / evaluate
        final_logs = train(
            epochs=num_epochs,
            model=model,
            model_name=f"{model_name}_{arch_type}_{emb_name}_{optim_name}",
            arch_type=arch_type,
            device=device,
            dataset=real_dataset,
            base_loss_fn=loss_fn,
            optimizer=optimizer,
            patience=patience
        )

        clean_up_model(model, optimizer)

        # Collect metrics
        if task_type == "regression":
            # Convert MSE → RMSE for comparability
            test_best_losses.append(np.sqrt(final_logs["test_best_loss"]))
        else:
            test_best_losses.append(final_logs["test_best_loss"])
            test_best_accuracies.append(final_logs["test_best_acc"])

        val_epoch_times.append(final_logs["val_epoch_time"])
        train_epoch_times.append(final_logs["train_epoch_time"])
        train_full_times.append(final_logs["num_epochs"] * final_logs["train_epoch_time"])
        best_test_epochs.append(final_logs["test_best_epoch"])
        train_epochs.append(final_logs["num_epochs"])
        num_params = final_logs["num_params"]

    # Select metric direction
    if task_type == "regression":
        metrics = test_best_losses
        direction = "min"
    else:
        metrics = test_best_accuracies
        direction = "max"

    # Hyperparameter snapshot
    hyperparams = collect_hyperparams(best_params, model_name, emb_name)

    # Aggregate statistics
    stats: Dict[str, Any] = {
        "num_params": num_params,
        # Epochs
        "num_epochs": np.mean(train_epochs),
        "num_epochs_std": np.std(train_epochs),
        "test_best_epoch": np.mean(best_test_epochs),
        "test_best_epoch_std": np.std(best_test_epochs),
        # Timing
        "train_epoch_time": np.mean(train_epoch_times),
        "train_epoch_time_std": np.std(train_epoch_times),
        "full_train_time": np.mean(train_full_times),
        "full_train_time_std": np.std(train_full_times),
        # Validation inference
        "val_epoch_time": np.mean(val_epoch_times),
        "val_epoch_time_std": np.std(val_epoch_times),
        # Losses
        "test_best_loss": np.mean(test_best_losses),
        "test_best_loss_std": np.std(test_best_losses),
        # Main metric
        "metric": np.mean(metrics),
        "metric_std": np.std(metrics),
        "direction": direction,
        # Dataset info
        "num_samples": dataset["info"]["num_samples"],
        "in_features": dataset["info"]["in_features"],
    }

    stats.update(hyperparams)
    return stats


