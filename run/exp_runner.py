import sys
import os
from pathlib import Path

HOME = str(Path.home())
sys.path.append(os.path.join(HOME, 'KAN', 'testing-kan'))

import torch.nn as nn
import numpy as np
from typing import Literal, Optional
import pandas as pd
import numpy as np
import pickle
import wandb
from IPython.display import clear_output
wandb.login(key='936d887040f82c8da3d87f5207c4178259c7b922')

from project_utils import utils
from project_utils import datasets
from project_utils.tg_bot import send_telegram_file, send_telegram_message
from project_utils.wandb_tuning import wandb_tuning
from project_utils.wandb_testing import test_best_model

from functools import wraps
import traceback
from datetime import datetime

def telegram_error_notification(func):
    """Декоратор для отправки ошибок в Telegram"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            
            # Формируем подробное сообщение
            error_msg = (
                f"*❌ Ошибка*\n\n"
                f"*Model:* `{kwargs.get('model_name', 'N/A')}`\n"
                f"*Dataset:* `{kwargs.get('dataset_name', 'N/A')}`\n"
                f"*Arch type:* `{kwargs.get('arch_type', 'N/A')}`\n"
                f"*Embedding:* `{kwargs.get('emb_name', 'N/A')}`\n"
                f"*Optimizer:* `{kwargs.get('optim_name', 'N/A')}`\n"
                f"*Функция:* `{func.__name__}`\n\n"
                f"*Ошибка:*\n```\n{str(e)}\n```\n\n"
                f"*Traceback:* (нажмите ▶️)\n"
                f"```\n{traceback.format_exc()}\n```"
            )
            send_telegram_message(error_msg)
            raise  # Пробрасываем исключение дальше
    return wrapper


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
    zip_path = os.path.join(HOME, 'KAN', 'data', f'{dataset_name}.zip')
    dataset = datasets.load_dataset(dataset_name, zip_path)
    pkl_logs = {}

    for model_name in model_names:
        for arch_type in arch_types:
            for optim_name in optim_names:
                for emb_name in emb_names:
                    stats = run_single_model(project_name, dataset_name, model_name, arch_type, emb_name, optim_name, dataset, num_epochs, num_trials, patience)
                    clear_output(wait=True)
                    pkl_logs[f'{model_name}_{arch_type}_{emb_name}_{optim_name}'] = stats
                    # send_telegram_message(f'✅ {model_name}_{arch_type}_{emb_name}_{optim_name} finished on {dataset_name}\
                    #                       Test sweep_id {stats["sweep_id"]}')

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
    send_telegram_message(
        f"🚀 *Запуск эксперимента*\n"
        f"▫️ *Название:* `{exp_name}`\n"
        f"▫️ *Время:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
    )
    total_logs = {}
    results_dir = os.path.join(HOME, 'KAN', 'testing-kan', 'results')

    # Создаем директорию для результатов, если ее нет
    os.makedirs(results_dir, exist_ok=True)
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
        # send_telegram_message(f'✅ Finished on {dataset_name}')

    # Сохраняем результаты с универсальным путем
    results_path = os.path.join(results_dir, f'{exp_name}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(total_logs, f)

    # send_telegram_message(f'✅ Finished {exp_name}')
    send_telegram_file(results_path)
    return total_logs

