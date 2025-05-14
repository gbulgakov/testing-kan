import shutil
import zipfile
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import pickle
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from optimizers.ademamix import AdEMAMix
from optimizers.muon import Muon
from optimizers.mars import MARS

# удобно для масшатбирования
OPTIMIZERS = { 
              'adamw' : torch.optim.AdamW,
              'ademamix' : AdEMAMix,
              'mars' : MARS,
              'muon' : Muon
             }


def seed_everything(seed=0):
    import random
    random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# True, если эпоха 1 лучше эпохи 2
def compare_epochs(task_type, epoch_dict1, epoch_dict2):
    if task_type == 'regression':
        return epoch_dict1['loss'] < epoch_dict2['loss']
    else:
        return epoch_dict1['acc'] > epoch_dict2['acc']
    

def get_sweep_config(model_name, emb_name, task_type, sweep_name):
    metric = {}
    if task_type == 'regression':
        metric = {'name' : 'val_best_loss', 'goal' : 'minimize'}
    else:
        metric = {'name' : 'val_best_acc', 'goal' : 'maximize'}
    params = {
        'lr' : {
            'distribution' : 'log_uniform_values',
            'min' : 1e-5,
            'max' : 1e-1 # потенциально могут подойти большие значения
        },
        'weight_decay' : {
            'distribution' : 'log_uniform_values',
            'min' : 1e-6,
            'max' : 5e-2
        }
    }
    if model_name == 'kan':
        params.update({
            'kan_layers' : {'values' : [1, 2, 3, 4, 5, 6]},   # грубый тюнинг
            'kan_width' : {'values' : [2 ** i for i in range(8)]},
            'grid_size' : {'values' : [i for i in range(3, 30, 2)]}
        })
    elif model_name == 'fast_kan': # RBF-KAN
        params.update({
            'kan_layers' : {'values' : [1, 2, 3, 4, 5, 6]},   # скрытые слои
            'kan_width' : {'values' : [2 ** i for i in range(1, 8)]}, 
            'grid_size' : {'values' : [i for i in range(4, 40, 2)]} # пусть будут четные
        })
    elif model_name == 'mlp':
        params.update({
            'mlp_layers' : {'values' : [1, 2, 3, 4]}, # скрытые слои
            'mlp_width' : {'values' : [2 ** i for i in range(1, 11)]},
            'use_dropout' : {'values' : [True, False], 'probabilities': [0.7, 0.3]},
            'dropout' : {'values' : [i / 100 for i in range(0, 55, 5)]}
        })
    config = {
        'method' : 'random',
        'metric' : metric,
        'parameters' : params,
        'name' : sweep_name
    }
    return config

def get_test_config(task_type, sweep_name):
    metric = {} # чисто технический параметр
    if task_type == 'regression':
        metric = {'name' : 'val_best_loss', 'goal' : 'minimize'}
    else:
        metric = {'name' : 'val_best_acc', 'goal' : 'maximize'}
    params = {
        'seed' : {
            'values' : [i for i in range(10)]
        }
    }
    config = {
        'method' : 'grid',
        'metric' : metric,
        'parameters' : params,
        'name' : sweep_name
    }
    return config

def get_optimizer(optim_name, model_params, config):
    optim_class = OPTIMIZERS[optim_name]
    optim_kwargs = {'lr' : config['lr']}
    if optim_name != 'muon': # это на будущее
        optim_kwargs['weight_decay'] = config['weight_decay']
    return optim_class(model_params, **optim_kwargs)
 
