from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

HOME = str(Path.home())
sys.path.append(os.path.join(HOME, "KAN", "testing-kan"))

from IPython.display import clear_output

from src.testing_kan import data_processing
from src.testing_kan.logger import Logger
from src.testing_kan.pipelines.testing import test_best_model
from src.testing_kan.pipelines.tuning import tune
from src.testing_kan.utils import create_zip_archive


def run_single_model(
    *,
    model_name: str,
    arch_type: str,
    emb_name: str,
    optim_name: str,
    dataset: Dict[str, Any],
    num_epochs: int,
    num_trials: int,
    patience: int,
) -> Dict[str, Any]:
    """Tune hyper-parameters, retrain the best model on 10 seeds and return stats."""
    best_params = tune(
        model_name=model_name,
        arch_type=arch_type,
        emb_name=emb_name,
        optim_name=optim_name,
        dataset=dataset,
        num_epochs=num_epochs,
        num_trials=num_trials,
        patience=patience,
    )

    clear_output(wait=True)

    stats = test_best_model(
        best_params=best_params,
        model_name=model_name,
        arch_type=arch_type,
        emb_name=emb_name,
        optim_name=optim_name,
        dataset=dataset,
        num_epochs=num_epochs,
        patience=patience,
    )
    return stats


def run_single_dataset(
    *,
    dataset_name: str,
    optim_names: List[str],
    emb_names: List[str],
    model_names: List[str],
    arch_types: List[str],
    num_epochs: int,
    num_trials: int,
    patience: int,
    logger: Logger,
) -> None:
    """Run the full grid of (model × arch × emb × optim) for one dataset."""
    zip_path = os.path.join(HOME, "KAN", "data", f"{dataset_name}.zip")
    dataset = data_processing.load_dataset(dataset_name, zip_path)

    for model_name in model_names:
        for arch_type in arch_types:
            for optim_name in optim_names:
                for emb_name in emb_names:
                    stats = run_single_model(
                        model_name=model_name,
                        arch_type=arch_type,
                        emb_name=emb_name,
                        optim_name=optim_name,
                        dataset=dataset,
                        num_epochs=num_epochs,
                        num_trials=num_trials,
                        patience=patience,
                    )

                    clear_output(wait=True)

                    logger.log_run(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        arch_type=arch_type,
                        emb_name=emb_name,
                        optim_name=optim_name,
                        stats=stats,
                    )


def run_experiment(
    *,
    dataset_names: List[str],
    model_names: List[str],
    emb_names: List[str],
    optim_names: List[str],
    arch_types: List[str],
    num_epochs: int,
    num_trials: int,
    patience: int,
    exp_name: str,
) -> None:
    """High-level entry point used by CLI (`run.py`)."""
    results_dir = os.path.join(HOME, "KAN", "testing-kan", "results", exp_name)
    logger = Logger(results_dir, exp_name)

    for dataset_name in dataset_names:
        run_single_dataset(
            dataset_name=dataset_name,
            optim_names=optim_names,
            emb_names=emb_names,
            model_names=model_names,
            arch_types=arch_types,
            num_epochs=num_epochs,
            num_trials=num_trials,
            patience=patience,
            logger=logger,
        )

    logger.save()

    # Zip all logs for easy sharing / upload
    archive_path = os.path.join(results_dir, f"{exp_name}_logs.zip")
    create_zip_archive(source_dir=results_dir, archive_path=archive_path)