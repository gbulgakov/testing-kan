from typing import Dict, Any

import torch
import wandb

from project_utils.utils import get_optimizer, get_sweep_config, get_test_config, seed_everything, clean_up_model
from project_utils.train import train, validate
from project_utils.datasets import get_dataloaders
from models.prepare_model import model_init_preparation, ModelWithEmbedding
from models.tabm_reference import Model


def wandb_tuning(project_name, dataset_name, 
                 model_name, arch_type, emb_name, optim_name, 
                 dataset, num_epochs=10, num_trials=30, patience=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_info = dataset['info']
    '''
    num_cont_cols = dataset['train']['X_num'].shape[1]
    X_cat = dataset['train'].get('X_cat', None)
    num_cat_cols = (X_cat.shape[1] if X_cat != None else 0)]
    '''
    num_cont_cols = dataset_info['num_cont_cols']
    num_cat_cols = dataset_info['num_cat_cols']
    sweep_name = f'tuning {model_name}_{arch_type}_{emb_name}_{optim_name} on {dataset_name}'

    # просто оборачиваем нашу train
    def sweep_wrapper():
        with wandb.init(
            project=f'{project_name}',
            group=f'dataset_{dataset_name}',
            tags=[f'model_{model_name}', f'arch_{arch_type}', f'emb_{emb_name}', 
                  f'optim_{optim_name}', f'dataset_{dataset_name}', 'tuning'],
            config=sweep_config
        ) as run:
            config = wandb.config
            _, layer_kwargs, backbone, bins, embeddings_kwargs, loss_fn, k = model_init_preparation(
                config=config,
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
            optimizer=get_optimizer(optim_name, model.parameters(), config)
            real_dataset = get_dataloaders(
                dataset=dataset,
                model=model,
                device=device,
                num_workers=4)
            real_dataset['info'] = dataset['info']
            train(
                epochs=num_epochs,
                model=model,
                model_name=f'{model_name}_{arch_type}_{emb_name}_{optim_name}',
                arch_type=arch_type,
                device=device,
                dataset=real_dataset,
                base_loss_fn=loss_fn,
                optimizer=optimizer,
                patience=patience
            )
          
            # очищаем память
            clean_up_model(model, optimizer)
    sweep_config = get_sweep_config(model_name, emb_name, dataset_info['task_type'], 
                                    f'tuning {model_name}_{arch_type}_{emb_name}_{optim_name} on {dataset_name}')
    
    sweep_id = wandb.sweep(sweep=sweep_config,
                           project=f'{project_name}',
                           entity='georgy-bulgakov') 
    wandb.agent(sweep_id, sweep_wrapper, count=num_trials)
    return sweep_id
