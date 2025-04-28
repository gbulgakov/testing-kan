import time
from typing import Dict, Any, List

import numpy as np
import torch
import wandb

from utils.utils import get_optimizer, get_sweep_config, get_test_config, seed_everything
from utils.train import train, validate
from models.prepare_model import model_init_preparation, ModelWithEmbedding, MLP
from models.tabm_reference import Model


def test_best_model(best_params, project_name, dataset_name, model_name, arch_type, emb_name, optim_name, dataset, num_epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_cont_cols = dataset['train']['X_num'].shape[1]
    X_cat = dataset['train'].get('X_cat', None)
    num_cat_cols = (X_cat.shape[1] if X_cat != None else 0)
    d_embedding = best_params.get('d_embedding', None)
    sigma = best_params.get('sigma', None)

    # подготовка логирования
    test_accuracies = []
    test_losses = []
    val_times = []
    train_times = []
    best_epochs = []

    testing_config = get_test_config(dataset['info']['task_type'], 
                                     f'testing {model_name}_{arch_type}_{emb_name}_{optim_name} on {dataset_name}')
    # обертка тестирования
    def test_wrapper():
        with wandb.init(
            project=f'{project_name}',
            group=f'dataset_{dataset_name}',
            tags=[f'model_{model_name}', f'arch_{arch_type}', f'emb_{emb_name}', 
                  f'optim_{optim_name}', f'dataset_{dataset_name}', 'testing'],
            config=testing_config
        ) as run:
            config = wandb.config
            # seed + подготовка модели
            seed_everything(config['seed'])
            _, backbone, bins, embeddings_kwargs, loss_fn, k = model_init_preparation(
                config=best_params,
                dataset=dataset,
                model_name=model_name,
                arch_type=arch_type,
                emb_name=emb_name
            )
            model = Model(
                n_num_features=num_cont_cols,
                n_cat_features=num_cat_cols,
                backbone=backbone,
                bins=bins,
                num_embeddings=embeddings_kwargs,
                arch_type=arch_type,
                k=k
            )
            full_train_time, val_time, test_loss, test_acc, test_best_epoch = train(
                epochs=num_epochs,
                model=model,
                model_name=f'{model_name}_{arch_type}_{emb_name}_{optim_name}',
                arch_type=arch_type,
                device=device,
                dataset=dataset,
                base_loss_fn=loss_fn,
                optimizer=get_optimizer(optim_name, model.parameters(), best_params)
            )
            test_loss, test_acc, test_time = validate(model, device, dataset, loss_fn, 'test', model_name, arch_type)
            # Логируем результаты для каждого сида
            logs = {
                'test_loss': test_loss,
                'test_best_epoch' : test_best_epoch,
                'full_train_time': full_train_time,
                'val_time': val_time,
                'seed': config['seed']
            }

            if dataset['info']['task_type'] != 'regression':
                logs.update({'test_acc' : test_acc})
            
            run.log(logs)
            test_accuracies.append(test_acc)
            if dataset['info']['task_type'] == 'regression':
                test_losses.append(np.sqrt(test_loss)) # переходим к RMSE и делаем обратное преобразование
            else:
                test_losses.append(test_loss)
            val_times.append(val_time)
            train_times.append(full_train_time)
            best_epochs.append(test_best_epoch)

    test_id = wandb.sweep(sweep=testing_config,
                           project=f'{project_name}',
                           entity='georgy-bulgakov') 
    wandb.agent(test_id, test_wrapper)

    # Создаем финальный run для агрегированных результатов
    with wandb.init(
        project=f"{project_name}",
        group=f'dataset_{dataset_name}',
        name="aggregated_results",
        tags=[f'model_{model_name}', f'arch_{arch_type}', f'emb_{emb_name}', f'optim_{optim_name}', f'dataset_{dataset_name}', 'testing'],
        config=best_params
    ) as run:
        stats = {
            'model' : f'{model_name}_{arch_type}_{emb_name}_{optim_name}',
            'mean_test_acc': np.mean(test_accuracies),
            'std_test_acc': np.std(test_accuracies),
            'mean_test_loss': np.mean(test_losses),
            'std_test_loss': np.std(test_losses),
            'mean_val_time': np.mean(val_times),
            'mean_train_time': np.mean(train_times),
            'mean_best_epoch' : np.mean(best_epochs),
            'std_best_epoch' : np.std(best_epochs),
            'all_test_accs' : test_accuracies,
            'all_test_losses' : test_losses,
            'all_val_times' : val_times,
            'all_train_times' : train_times,
            'all_best_epochs' : best_epochs
        }
        
        run.log(stats)
        stats['name'] = f'{model_name}_{emb_name}_{optim_name}'
        
    keys = ['model', 'mean_test_acc', 'std_test_acc', 'mean_test_loss', 'std_test_loss', 'mean_test_time', 'mean_train_time']
    return {key : stats[key] for key in keys} # чисто технически для удобства
