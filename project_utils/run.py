import sys
import torch.nn as nn
import numpy as np
from typing import Literal, Optional
import pandas as pd
import numpy as np
import torch
import pickle
import wandb
from IPython.display import clear_output
wandb.login(key='936d887040f82c8da3d87f5207c4178259c7b922')

from models.efficient_kan import KAN
from models.fastkan import FastKAN
from models.chebyshev_kan import ChebyKAN
from models.mlp import MLP
from models.prepare_model import model_init_preparation, ModelWithEmbedding

from project_utils import utils
from project_utils import datasets
from project_utils.tg_bot import send_telegram_file, send_telegram_message
from project_utils.wandb_tuning import wandb_tuning
from project_utils.wandb_testing import test_best_model

def run_single_model(project_name, dataset_name, model_name, arch_type, emb_name, optim_name, dataset, num_epochs, num_trials, patience):
    sweep_id = wandb_tuning(project_name, dataset_name, model_name, arch_type, emb_name, optim_name, dataset, num_epochs, num_trials, patience)
    clear_output(wait=True)
    # вспоминаем лучшие параметры
    api = wandb.Api()
    sweep = api.sweep(f'georgy-bulgakov/{project_name}/{sweep_id}')
    runs = sweep.runs
    best_run = None

    def safe_score(score, direction):
        if isinstance(score, str) or score is None:
            return 1e20 if direction == 'minimize' else 0
        return score

    if dataset['info']['task_type'] == 'regression':
        best_run = min(runs, 
                       key=lambda run : safe_score(run.summary.get('val_loss'), 'minimize'))
    else:
        best_run = max(runs, 
                       key=lambda run : safe_score(run.summary.get('val_acc'), 'maximize'))
    
    best_params = best_run.config
    stats = test_best_model(best_params, project_name, dataset_name, model_name, arch_type, emb_name, optim_name, dataset, num_epochs, patience)
    return stats, best_params

def run_single_dataset(project_name, dataset_name, 
                       optim_names, emb_names, model_names, arch_types,
                       num_epochs, num_trials, patience):
    # dataset_type = dataset_info['type']
    zip_path = f'data/{dataset_name}.zip'
    dataset = datasets.load_dataset(dataset_name, zip_path)
    results = []
    pkl_logs = {}

    for model_name in model_names: # можно оставить только kan, тогда model_names = ['kan']
        for arch_type in arch_types:
            for optim_name in optim_names:
                for emb_name in emb_names:
                    stats, best_params = run_single_model(project_name, dataset_name, model_name, arch_type, emb_name, optim_name, dataset, num_epochs, num_trials, patience)
                    results.append(stats)
                    clear_output(wait=True)
                    pkl_logs[f'{model_name}_{arch_type}_{emb_name}_{optim_name}'] = (stats | best_params)
                    send_telegram_message(f'✅ {model_name}_{arch_type}_{emb_name}_{optim_name} finished on {dataset_name}')

    
    with wandb.init(
        project=project_name,
        group=f'dataset_{dataset_name}',
        name='models_comparison'
    ) as run:
        run.log({
            f'final_table_{dataset_name}' : wandb.Table(dataframe=pd.DataFrame(results))
        })

    with open(f'results/{dataset_name}.pkl', 'wb') as f:
        pickle.dump(pkl_logs, f)

    send_telegram_file(f'results/{dataset_name}.pkl')
    return results


def run_experiment(
    *,
    project_name,
    dataset_names,
    model_names,
    emb_names,
    optim_names,
    arch_types,
    num_epochs,
    num_trials,
    patience
):
    for dataset_name in dataset_names:
        run_single_dataset(
            project_name=project_name,
            dataset_name=dataset_name,
            optim_names=optim_names,
            emb_names=emb_names,
            model_names=model_names,
            arch_types=arch_types,
            num_epochs=num_epochs,
            num_trials=num_trials,
            patience=patience
        )

if __name__ == '__main__':
    run_experiment(
        project_name='Random ablation study',
        dataset_names=['sberbank-housing', 'homesite-insurance', 'facebook', 'house', 'gesture', 'churn', 'california', 'adult'],
        model_names=['kan', 'fast_kan'],
        emb_names=['none'],
        optim_names=['adamw'],
        arch_types=['plain'],
        num_epochs=100,
        num_trials=200,
        patience=5
    )
