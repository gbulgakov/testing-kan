from logging import config
from random import Random
from typing import Dict, Any
import optuna
from optuna.samplers import RandomSampler
import time
import torch
import wandb

from gbdt_utils import get_sweep_config, model_init_preparation, suggest_params
from src.testing_kan.utils import clean_up_model


def train(dataset, model, fit_kwargs, predict):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_time = time.time()
    model.fit(dataset['train']['X'], dataset['train']['y'], **fit_kwargs)
    end_time = time.time()
    task_type = dataset['info']['task_type']
    start_val_time = time.time()
    val_predict = predict(dataset['val']['X'])
    end_val_time = time.time()
    test_predict = predict(dataset['test']['X'])

    final_logs = {
        'train_time': end_time - start_time,
        'val_time': end_val_time - start_val_time,
        'in_features': dataset['info']['in_features'],
        'out_features': (1 if task_type != 'multiclass' else dataset['info']['n_classes']),
    }
    if task_type == 'regression':
        val_loss = ((dataset['val']['y'] - val_predict) ** 2).mean()
        test_loss = ((dataset['test']['y'] - test_predict) ** 2).mean()
        final_logs['val_loss'] = val_loss
        final_logs['test_loss'] = test_loss
    else:
        if task_type == 'multiclass':
            val_pred = val_predict.argmax(1)
            test_pred = test_predict.argmax(1)
        if task_type == 'binclass':
            val_pred = (val_predict >= 0)
            test_pred = (test_predict >= 0)
        val_correct = (val_pred == dataset['val']['y']).sum()
        test_correct = (test_pred == dataset['test']['y']).sum()
        val_accuracy = val_correct / dataset['val']['y'].size
        test_accuracy = test_correct / dataset['test']['y'].size
        final_logs['val_accuracy'] = val_accuracy
        final_logs['test_accuracy'] = test_accuracy
    return final_logs

def make_objective(
        *,
        model_name,
        dataset,
):
    def objective(trial):
        dataset_info = dataset['info']
        task_type = dataset_info['task_type']
        metric = 'val_loss' if task_type == 'regression' else 'val_accuracy'
        hyperparams = suggest_params(trial, model_name, task_type=task_type)
        model, fit_kwargs, predict = model_init_preparation(
            config=hyperparams,
            dataset=dataset,
            model_name=model_name
        )
        final_logs = train(
            dataset=dataset,
            model=model,
            fit_kwargs=fit_kwargs,
            predict=predict
        )
        clean_up_model(model)
        return final_logs[metric]
    
    return objective

def tune(
        model_name,
        dataset,
        num_trials=70
):
    dataset_info = dataset['info']
    task_type = dataset_info['task_type']
    direction = 'minimize' if task_type == 'regression' else 'maximize'
    
    objective = make_objective(
        model_name=model_name,
        dataset=dataset
    )
    study = optuna.create_study(
        direction=direction,
        sampler=RandomSampler()
    )

    study.optimize(
        objective,
        n_trials=num_trials
    )

    return study.best_params

# def wandb_tuning(project_name, dataset_name, 
#                  model_name, dataset, num_trials):
#     dataset_info = dataset['info']
#     task_type = dataset_info['task_type']
#     num_cont_cols = dataset_info['num_cont_cols']
#     num_cat_cols = dataset_info['num_cat_cols']
#     sweep_name = f'tuning {model_name} on {dataset_name}'

#     def sweep_wrapper():
#         with wandb.init(
#             project=f'{project_name}',
#             group=f'dataset_{dataset_name}',
#             tags=[f'model_{model_name}', f'dataset_{dataset_name}', 'tuning'],
#             config=sweep_config
#         ) as run:
#             config = wandb.config
#             model, fit_kwargs, predict = model_init_preparation(
#                 config=config,
#                 dataset=dataset,
#                 model_name=model_name,
#             )
#             final_logs = train(dataset, model, fit_kwargs, predict)
#             wandb.log(final_logs)


#     sweep_config = get_sweep_config(model_name, dataset_info['task_type'], 
#                                     sweep_name=sweep_name)
    
#     sweep_id = wandb.sweep(sweep=sweep_config,
#                            project=f'{project_name}',
#                            entity='georgy-bulgakov')
#     wandb.agent(sweep_id, sweep_wrapper, count=num_trials)
#     return sweep_id
