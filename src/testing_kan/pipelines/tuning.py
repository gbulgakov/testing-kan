from typing import Dict, Any, Callable
import torch
import optuna
from optuna.samplers import RandomSampler
from src.testing_kan.utils import get_optimizer, clean_up_model, seed_everything
from .train import train
from src.testing_kan.data_processing import get_dataloaders
from src.testing_kan.models.prepare_model import model_init_preparation
from src.testing_kan.models.tabm_reference import Model


def suggest_params(trial: optuna.Trial, model_name: str, emb_name: str) -> Dict[str, Any]:
    """
    Suggest hyperparameters for the given model and embedding configuration.
    """
    params = {}
    
    # Common parameters for all models
    params['lr'] = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    
    # Model-specific parameters
    if model_name == 'mlp':
        params['mlp_layers'] = trial.suggest_int('mlp_layers', 1, 4)
        params['mlp_width'] = trial.suggest_categorical('mlp_width', [2 ** i for i in range(1, 11)])
        
        # Dropout with 70% probability
        use_dropout = trial.suggest_float('use_dropout_decision', 0.0, 1.0) < 0.7
        if use_dropout:
            params['dropout'] = trial.suggest_float('dropout', 0.05, 0.5, step=0.05)
        else:
            params['dropout'] = trial.suggest_float('dropout', 0.0, 0.0)  # No dropout
        
    elif model_name == 'kan':       # for ablation studies
        params['kan_layers'] = trial.suggest_int('kan_layers', 1, 5)
        params['kan_width'] = trial.suggest_categorical('kan_width', [2 ** i for i in range(1, 8)])
        params['grid_size'] = trial.suggest_int('grid_size', 3, 29, step=2)
        
    elif model_name == 'small_kan':  # Smaller KAN with constrained parameter space (basic model)
        params['kan_layers'] = trial.suggest_int('kan_layers', 1, 3)
        params['kan_width'] = trial.suggest_int('kan_width', 16, 64, step=8)
        params['grid_size'] = trial.suggest_int('grid_size', 5, 15, step=2)

    elif model_name == 'fast_kan': # RBF-KAN
        params['kan_layers'] = trial.suggest_int('kan_layers', 1, 5)
        params['kan_width'] = trial.suggest_categorical('kan_width', [2 ** i for i in range(3, 8)])
        params['grid_size'] = trial.suggest_int('grid_size', 6, 16, step=2)  

    elif model_name == 'cheby_kan': # Chebyshev-KAN
        params['kan_layers'] = trial.suggest_int('kan_layers', 1, 5)
        params['kan_width'] = trial.suggest_categorical('kan_width', [2 ** i for i in range(3, 8)])
        params['degree'] = trial.suggest_int('degree', 1, 14) 

    # Embedding parameters
    if emb_name != 'none':
        params['d_embedding'] = trial.suggest_categorical('d_embedding', [2 ** i for i in range(1, 8)])
        
        if emb_name == 'periodic':
            params['sigma'] = trial.suggest_float('sigma', 0.01, 100, log=True)
            
        if emb_name == 'kan_emb':
            params['emb_grid_size'] = trial.suggest_int('emb_grid_size', 3, 15, step=2)
    
    return params

def make_objective(
    *,
    device: torch.device,
    model_name: str,
    arch_type: str,
    emb_name: str,
    optim_name: str,
    dataset: Dict[str, Any],
    num_epochs: int = 100,
    patience: int = 5,
) -> Callable[[optuna.Trial], float]:
    """
    Create an objective function for hyperparameter optimization.
    """
    def objective(trial: optuna.Trial) -> float:
        seed_everything(52)

        # Extract dataset information
        dataset_info = dataset['info']
        num_cont_cols = dataset_info['num_cont_cols']
        num_cat_cols = dataset_info['num_cat_cols']

        # Choose optimization metric based on task type
        metric = 'val_best_loss' if dataset_info['task_type'] == 'regression' else 'val_best_acc'

        # Sample hyperparameters
        hyperparams = suggest_params(trial, model_name, emb_name)
        
        # Initialize model components
        _, layer_kwargs, backbone, bins, embeddings_kwargs, loss_fn, k = model_init_preparation(
            config=hyperparams,
            dataset=dataset,
            model_name=model_name,
            arch_type=arch_type,
            emb_name=emb_name
        )
        
        # Build model
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
        
        # Initialize optimizer
        optimizer = get_optimizer(optim_name, model.parameters(), hyperparams)
        
        # Prepare data loaders
        real_dataset = get_dataloaders(
            dataset=dataset,
            model=model,
            device=device,
            num_workers=4
        )
        real_dataset['info'] = dataset['info']
        
        # Train model
        final_logs = train(
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
            
        # Clean up resources
        clean_up_model(model, optimizer)
        
        return final_logs[metric]

    return objective

def tune(
    model_name: str,
    arch_type: str,
    emb_name: str,
    optim_name: str,
    dataset: Dict[str, Any],
    num_epochs: int = 100,
    num_trials: int = 70,
    patience: int = 5
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning using Optuna.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset_info = dataset['info']
    task_type = dataset_info['task_type']
    
    # Set optimization direction based on task type
    direction = 'minimize' if task_type == 'regression' else 'maximize'
    
    # Create objective function
    objective = make_objective(
        device=device,
        model_name=model_name,
        arch_type=arch_type,
        emb_name=emb_name,
        optim_name=optim_name,
        dataset=dataset,
        num_epochs=num_epochs,
        patience=patience
    )
    
    # Create and run optimization study
    study = optuna.create_study(
        direction=direction,
        sampler=RandomSampler()
    )
    
    study.optimize(
        objective,
        n_trials=num_trials
    )
    
    return study.best_params