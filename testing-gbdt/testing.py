import time
from typing import Dict, Any, List

import numpy as np
import torch
import wandb

from gbdt_utils import get_sweep_config, get_test_config, seed_everything, count_parameters, model_init_preparation
from tuning import train

def collect_hyperparams(best_params, model_name, emb_name):
    if model_name == 'catboost':
        hyperparams = {
            'learning_rate' : best_params.get('learning_rate'),
            'bagging_temperature' : best_params.get('bagging_temperature'),
            'depth' : best_params.get('depth'),
            'l2_leaf_reg' : best_params.get('l2_leaf_reg'),
            'leaf_estimation_iterations' : best_params.get('leaf_estimation_iterations'),
        }
    elif model_name == 'xgboost':
        hyperparams = {
            'learning_rate' : best_params.get('learning_rate'),
            'colsample_bytree' : best_params.get('colsample_bytree'),
            'colsample_bylevel' : best_params.get('colsample_bylevel'),
            'gamma' : best_params.get('gamma'),
            'lambda' : best_params.get('lambda'),
            'max_depth' : best_params.get('max_depth'),
            'min_child_weight' : best_params.get('min_child_weight'),
            'subsample' : best_params.get('subsample'),
        }
    elif model_name == 'lightgbm':
        hyperparams = {
            'learning_rate' : best_params.get('learning_rate'),
            'num_leaves' : best_params.get('num_leaves'),
            'min_child_weight' : best_params.get('min_child_weight'),
            'min_child_samples' : best_params.get('min_child_samples'),
            'subsample' : best_params.get('subsample'),
            'colsample_bytree' : best_params.get('colsample_bytree'),
            'reg_lambda' : best_params.get('reg_lambda'),
        }
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return hyperparams


import numpy as np

def test_best_model(
    best_params,
    project_name,
    dataset_name,
    model_name,
    dataset
):
    task_type = dataset['info']['task_type']

    # Списки для сбора метрик (как в оригинальном коде)
    test_accuracies = []
    test_losses = []
    val_times = []
    train_times = []

    # Цикл по 10 сидам (простой loop вместо sweep/agent)
    for seed in range(10):
        # Установка seed
        seed_everything(seed)
        
        # Инициализация модели с фиксированными best_params
        model, fit_kwargs, predict = model_init_preparation(
            config=best_params,
            dataset=dataset,
            model_name=model_name,
        )
        
        # Обучение
        final_logs = train(dataset, model, fit_kwargs, predict)
        
        # Сбор метрик
        if task_type == 'regression':
            test_losses.append(np.sqrt(final_logs.get('test_loss', 0)))  # RMSE для consistency с шаблоном; удалите sqrt, если не нужно
        else:
            test_losses.append(final_logs.get('test_loss', 0))
            test_accuracies.append(final_logs.get('test_accuracy', 0))
        
        val_times.append(final_logs.get('val_time', 0))
        train_times.append(final_logs.get('train_time', 0))
        
        # Простой logging (замена wandb.log; опционально)
        print(f"Seed {seed}: final_logs={final_logs}")

    # Ключевые метрики (как в оригинальном коде)
    if task_type == 'regression':
        metrics = test_losses
        direction = 'min'
    else:
        metrics = test_accuracies
        direction = 'max'
    
    # Гиперпараметры (через вашу функцию)
    hyperparams = collect_hyperparams(best_params, model_name, emb_name='none')

    # Аггрегированные stats (как в оригинальном коде)
    stats = {
        'model_name': model_name,
        'real_metric': np.mean(metrics),
        'metric_std': np.std(metrics),
        'direction': direction,
        'full_train_time': np.mean(train_times),
        'full_train_time_std': np.std(train_times),
        'full_val_time': np.mean(val_times),
        'full_val_time_std': np.std(val_times),
        # Нет sweep_id, так как нет sweep; можно добавить 'project_name' или 'dataset_name', если нужно
        'project_name': project_name,
        'dataset_name': dataset_name,
        # Размеры датасета
        'num_samples': dataset['info']['num_samples'],
        'in_features': dataset['info']['in_features']
    }
    stats.update(hyperparams)  # Добавляем гиперпараметры
    return stats