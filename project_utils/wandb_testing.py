import time
from typing import Dict, Any, List

import numpy as np
import torch
import wandb

from project_utils.utils import get_optimizer, get_sweep_config, get_test_config, seed_everything
from project_utils.train import train, validate
from models.prepare_model import model_init_preparation, ModelWithEmbedding, MLP
from models.tabm_reference import Model

def test_best_model(best_params, project_name, dataset_name, model_name, arch_type, emb_name, optim_name, dataset, num_epochs=10, patience=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task_type = dataset['info']['task_type']

    num_cont_cols = dataset['info']['num_cont_cols']
    num_cat_cols = dataset['info']['num_cat_cols']
    d_embedding = best_params.get('d_embedding', None)
    sigma = best_params.get('sigma', None)
    emb_grid_size = best_params.get('emb_grid_size', None)

    # подготовка логирования
    test_best_accuracies = []
    test_best_losses = []
    test_real_accuracies = []
    test_real_losses = []
    val_epoch_times = []
    train_epoch_times = []
    train_full_times = []
    train_epochs = []
    best_test_epochs = []

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
            _, layer_kwargs, backbone, bins, embeddings_kwargs, loss_fn, k = model_init_preparation(
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
                k=k,
                **layer_kwargs
            )
            total_epochs, train_epoch_time, val_epoch_time, \
            test_best_loss, test_best_acc, test_best_epoch, \
            test_real_loss, test_real_acc, test_real_epoch = train(
                epochs=num_epochs,
                model=model,
                model_name=f'{model_name}_{arch_type}_{emb_name}_{optim_name}',
                arch_type=arch_type,
                device=device,
                dataset=dataset,
                base_loss_fn=loss_fn,
                optimizer=get_optimizer(optim_name, model.parameters(), best_params),
                patience=patience
            )
            # Логируем результаты для каждого сида
            logs = {
                'test_best_loss': test_best_loss,
                'test_best_epoch' : test_best_epoch,
                'test_real_loss' : test_real_loss,
                'test_real_epoch' : test_real_epoch,
                'train_epoch_time': train_epoch_time,
                'val_epoch_time' : val_epoch_time,
                'full_train_time' : total_epochs * train_epoch_time,
                'seed': config['seed']
            }

            if task_type != 'regression':
                logs.update({
                    'test_best_acc' : test_best_acc,
                    'test_real_acc' : test_best_acc
                })
            
            run.log(logs)
            test_best_accuracies.append(test_best_acc)
            test_real_accuracies.append(test_real_acc)
            if dataset['info']['task_type'] == 'regression':
                test_best_losses.append(np.sqrt(test_best_loss)) # переходим к RMSE
                test_real_losses.append(np.sqrt(test_real_loss))
            else:
                test_best_losses.append(test_best_loss)
                test_real_losses.append(test_real_losses)
            
            val_epoch_times.append(val_epoch_time)
            train_epoch_times.append(train_epoch_time)
            train_full_times.append(total_epochs * train_epoch_time)
            best_test_epochs.append(test_best_epoch)
            train_epochs.append(total_epochs)

    test_id = wandb.sweep(sweep=testing_config,
                           project=f'{project_name}',
                           entity='georgy-bulgakov') 
    wandb.agent(test_id, test_wrapper)


    # ключевые метрики для удобства
    if dataset['info']['task_type'] == 'regression':
        metrics = test_best_losses
        real_metrics = test_real_losses
        direction = 'min'
    else:
        metrics = test_best_accuracies
        real_metrics = test_real_accuracies
        direction = 'max'
    
    # гиперпараметры отдельно для удобства
    if model_name == 'mlp':
        hparams = {
            'width' : best_params.get('mlp_width'),
            'hidden_layers' : best_params.get('mlp_layers'),
            'use_dropout' : best_params.get('use_dropout'),
            'dropout' : (0 if not best_params.get('use_dropout') else best_params.get('dropout'))
        }
    else:
        hparams = {
            'width' : best_params.get('kan_width'),
            'hidden_layers' : best_params.get('kan_layers'),
            'grid_size' : best_params.get('grid_size')
        }
    if emb_name != 'none':
        hparams.update({
            'd_embedding' : d_embedding
        })
        if emb_name == 'periodic':
            hparams.update({
                'sigma' : sigma
            })
        if emb_name in ['kan_emb', 'fast_kan_emb']:
            hparams.update({
                'emb_grid_size' : emb_grid_size
            })
    hparams.update({
        'lr' : best_params.get('lr'),
        'weight_decay' : best_params.get('weight_decay')
    })
    
    # Логируем аггрегированные результаты
    stats = {
        # модели
        'model' : f'{model_name}_{arch_type}_{emb_name}_{optim_name}',
        'model_name' : model_name,
        'arch_type' : arch_type,
        'emb_name' : emb_name,
        'optim_name' : optim_name,
        # эпохи
        'num_epochs' : np.mean(train_epochs),
        'num_epochs_std' : np.std(train_epochs),
        'test_best_epochs' : np.mean(best_test_epochs),
        'test_best_epochs_std' : np.std(best_test_epochs),
        # времена обучения
        'train_epoch_time' : np.mean(train_epoch_times),
        'train_epoch_time_std' : np.std(train_epoch_times),    
        'full_train_time' : np.mean(train_full_times),
        'full_train_time_std' : np.std(train_full_times),      
        # времена инференса (на val)
        'val_epoch_time' : np.mean(val_epoch_times),
        'val_epoch_time_std' : np.std(val_epoch_times),   
        # лоссы 
        'test_best_loss' : np.mean(test_best_losses),
        'test_best_loss_std' : np.std(test_best_losses),
        'test_real_loss' : np.mean(test_real_losses),
        'test_real_loss_std' : np.std(test_real_losses),
        # метрики (точность/лосс) (для удобства)
        'metric' : np.mean(metrics),
        'metric_std' : np.std(metrics),
        'real_metrics' : np.mean(real_metrics),
        'real_metrics_std' : np.std(real_metrics),
        'direction' : direction,
        # sweep id
        'sweep_id' : test_id,
        # размеры датасета
        'num_samples' : dataset['info']['num_samples'],
        'in_features' : dataset['info']['in_features']
    }
    stats.update(hparams)
    return stats # чисто технически для удобства