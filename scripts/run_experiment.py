import sys
import os
from pathlib import Path

HOME = str(Path.home())
sys.path.append(os.path.join(HOME, 'KAN', 'testing-kan'))

from IPython.display import clear_output

from src.testing_kan import data_processing
from src.testing_kan.pipelines.tuning import tune
from src.testing_kan.pipelines.testing import test_best_model
from src.testing_kan.logger import Logger
from src.testing_kan.utils import create_zip_archive



def run_single_model(
        *,
        model_name, 
        arch_type, 
        emb_name, 
        optim_name, 
        dataset, 
        num_epochs, 
        num_trials, 
        patience
):
    best_params = tune(
            model_name=model_name,
            arch_type=arch_type,
            emb_name=emb_name,
            optim_name=optim_name,
            dataset=dataset,
            num_epochs=num_epochs,
            num_trials=num_trials,
            patience=patience
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
        patience=patience
    )
    return stats

def run_single_dataset(
        *, 
        dataset_name,
        optim_names, 
        emb_names, 
        model_names, 
        arch_types,
        num_epochs, 
        num_trials, 
        patience, 
        logger
):
    # dataset_type = dataset_info['type']
    zip_path = os.path.join(HOME, 'KAN', 'data', f'{dataset_name}.zip')
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
                        patience=patience
                    )
                    clear_output(wait=True)
                    
                    logger.log_run(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        arch_type=arch_type,
                        emb_name=emb_name,
                        optim_name=optim_name,
                        stats=stats
                    )


def run_experiment(
    *,
    dataset_names,
    model_names,
    emb_names,
    optim_names,
    arch_types,
    num_epochs,
    num_trials,
    patience,
    exp_name
):
    # Logger
    results_dir = os.path.join(HOME, 'KAN', 'testing-kan', 'results', exp_name)
    logger = Logger(results_dir, exp_name)

    # runs
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
            logger=logger
        )

    logger.save()

    archive_name = f'{exp_name}_logs.zip'
    archive_path = os.path.join(results_dir, archive_name)
    create_zip_archive(source_dir=results_dir, archive_path=archive_path)

    

