import sys
sys.path.append('/home/no_prolactin/KAN/testing-kan')
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

    if dataset['info']['task_type'] == 'regression':
        best_run = min(runs, key=lambda run: run.summary.get('val_loss', float('inf')))
    else:
        best_run = max(runs, key=lambda run: run.summary.get('val_acc', 0))
    best_params = best_run.config

    stats = test_best_model(best_params, project_name, dataset_name, model_name, arch_type, emb_name, optim_name, dataset, num_epochs, patience)
    return stats

def run_single_dataset(project_name, dataset_name, 
                       optim_names, emb_names, model_names, arch_types,
                       num_epochs, num_trials, patience):
    # dataset_type = dataset_info['type']
    zip_path = f'data/{dataset_name}.zip'
    dataset = datasets.load_dataset(dataset_name, zip_path)
    pkl_logs = {}

    for model_name in model_names:
        for arch_type in arch_types:
            for optim_name in optim_names:
                for emb_name in emb_names:
                    stats = run_single_model(project_name, dataset_name, model_name, arch_type, emb_name, optim_name, dataset, num_epochs, num_trials, patience)
                    clear_output(wait=True)
                    pkl_logs[f'{model_name}_{arch_type}_{emb_name}_{optim_name}'] = stats
                    send_telegram_message(f'✅ {model_name}_{arch_type}_{emb_name}_{optim_name} finished on {dataset_name}\
                                          Test sweep_id {stats["sweep_id"]}')

    # with open(f'/home/no_prolactin/KAN/testing-kan/results/{dataset_name}.pkl', 'wb') as f:
    #     pickle.dump(pkl_logs, f)

    # send_telegram_file(f'/home/no_prolactin/KAN/testing-kan/results/{dataset_name}.pkl')
    return pkl_logs


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
    patience,
    exp_name
):
    total_logs = {}
    for dataset_name in dataset_names:
        logs = run_single_dataset(
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
        total_logs[dataset_name] = logs
        send_telegram_message(f'✅ Finished on {dataset_name}')

    with open(f'/home/no_prolactin/KAN/testing-kan/results/{exp_name}.pkl', 'wb') as f:
        pickle.dump(total_logs, f)
    send_telegram_message(f'✅ Finished {exp_name}')
    send_telegram_file(f'/home/no_prolactin/KAN/testing-kan/results/{exp_name}.pkl')
    return total_logs
    
        
        

if __name__ == '__main__':
    run_experiment(
        project_name='Embeddings 2.0 on GPU',
        dataset_names=['gesture', 'house', 'california'],
        model_names=['mlp', 'small_kan', 'kan', 'fast_kan'],
        emb_names=['none'],
        optim_names=['adamw'],
        arch_types=['plain'],
        num_epochs=100,
        num_trials=70,
        patience=5
    )
