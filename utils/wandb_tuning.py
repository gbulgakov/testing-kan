from typing import Dict, Any

import torch
import wandb

from utils.utils import get_optimizer, get_sweep_config, get_test_config, seed_everything
from utils.train import train, validate
from models.prepare_model import model_init_preparation, ModelWithEmbedding


def wandb_tuning(project_name, dataset_name, 
                 model_name, arch_type, emb_name, optim_name, 
                 dataset, num_epochs=10, num_trials=30):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_info = dataset['info']
    num_cont_cols = dataset['train']['X_num'].shape[1]
    sweep_name = f'tuning {model_name}_{emb_name}_{optim_name} on {dataset_name}'

    # просто оборачиваем нашу train
    def sweep_wrapper():
        with wandb.init(
            project=f'{project_name}',
            group=f'dataset_{dataset_name}',
            tags=[f'model_{model_name}', f'emb_{emb_name}', f'optim_{optim_name}', f'dataset_{dataset_name}', 'tuning'],
            config=sweep_config
        ) as run:
            config = wandb.config
            _, backbone, bins, loss_fn = model_init_preparation(
                config=config,
                dataset=dataset,
                model_name=model_name,
                emb_name=emb_name
            )
            model = ModelWithEmbedding(
                n_cont_features=num_cont_cols,
                d_embedding=config.get('d_embedding', None),
                emb_name=emb_name,
                backbone_model=backbone,
                bins=bins, 
                sigma=config.get('sigma', None)
            )
            train(
                epochs=num_epochs,
                model=model,
                model_name=f'{model_name}_{arch_type}_{emb_name}_{optim_name}',
                arch_type=arch_type,
                device=device,
                dataset=dataset,
                base_loss_fn=loss_fn,
                optimizer=get_optimizer(optim_name, model.parameters(), config),
            )
    sweep_config = get_sweep_config(model_name, emb_name, dataset_info['task_type'], 
                                    f'tuning {model_name}_{emb_name}_{optim_name} on {dataset_name}')
    
    sweep_id = wandb.sweep(sweep=sweep_config,
                           project=f'{project_name}',
                           entity='georgy-bulgakov') 
    wandb.agent(sweep_id, sweep_wrapper, count=num_trials)
    return sweep_id
