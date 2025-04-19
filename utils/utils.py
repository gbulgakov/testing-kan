import shutil
import zipfile
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import pickle
import json
import torch
import numpy as np

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

def load_dataset(name):
    zip_path = f'/kaggle/working/{name}.zip'
    data = {'train' : {}, 'val' : {}, 'test' : {}}
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Создаем временную директорию для распаковки
        temp_dir = Path(f'{name}_data')
        zip_ref.extractall(temp_dir)
        
        # Загружаем метаданные
        with open(temp_dir / f'{name}' / 'info.json') as f:
            data['info'] = json.load(f)

        # Для кат. фич
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Загружаем все .npy файлы
        for part in ['train', 'val', 'test']:
            for data_type in ['X_num', 'X_cat', 'y']:
                file_path = temp_dir / f'{name}' / f'{data_type}_{part}.npy'
                if file_path.exists():
                    data[part][data_type] = np.load(file_path, allow_pickle=True)

        # Обучение ohe на train и кодирование
            cat_path = temp_dir / f'{name}' / f'X_cat_{part}.npy'
            if cat_path.exists():
                if part == 'train':
                    one_hot_encoder.fit(data['train']['X_cat']) 
                data[part]['X_cat'] = one_hot_encoder.transform(data[part]['X_cat'])

        # переводим данные в тензоры
        for part in ['train', 'val', 'test']:
            for data_type in data[part].keys():
                data[part][data_type] = torch.tensor(data[part][data_type], dtype=torch.float)
        
        # Удаляем временную директорию
        shutil.rmtree(temp_dir)
    return data

import os
import pickle

def write_results(pkl_path, model_name, emb_name, optim_name,
                  layers, num_epochs, num_params, best_params, 
                  test_accuracies, test_loss, train_times, 
                  test_times, train_loss_history, val_loss_history):
    best_params['pkl_path'] = pkl_path
    best_params['layers'] = layers
    best_params['num_epochs'] = num_epochs
    best_params['num_params'] = num_params
    best_params['test_accuracies'] = test_accuracies
    best_params['test_loss'] = test_loss
    best_params['train_times'] = train_times
    best_params['test_times'] = test_times # т.к. на этапе тестирования нет обучения, то это по сути время инференса
    best_params['train_loss_history'] = train_loss_history
    best_params['val_loss_history'] = val_loss_history

    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        data[f'{optim_name}_{emb_name}_{model_name}'] = best_params
        
    else:
        data = {f'{optim_name}_{emb_name}_{model_name}' : best_params}    
    
    with open(pkl_path, 'wb') as file:
        pickle.dump(data, file)

def get_sweep_config(model_name, emb_name, task_type, sweep_name):
    metric = {}
    if task_type == 'regression':
        metric = {'name' : 'val_loss', 'goal' : 'minimize'}
    else:
        metric = {'name' : 'val_acc', 'goal' : 'maximize'}
    
    max_log_width = (7 if model_name == 'kan' else 11)
    params = {
        'lr' : {
            'distribution' : 'log_uniform_values',
            'min' : 1e-5,
            'max' : 1e-2
        },
        'weight_decay' : {
            'distribution' : 'log_uniform_values',
            'min' : 1e-6,
            'max' : 1e-3
        }
    }

    if model_name == 'mlp':
        params.update({
            'mlp_layers' : {'values' : [1, 2, 3, 4]}, # скрытые слои
            'mlp_width' : {'values' : [2 ** i for i in range(11)]},
             'use_dropout' : {'values' : [True, False],
                             'probabilities' : [0.7, 0.3] # dropout вероятно нужен
                            },
            'dropout' : {'values' : [i / 100 for i in range(0, 55, 5)]}
        })
    elif model_name == 'kan':
        params.update({
            'kan_layers' : {'values' : [1, 2, 3, 4]},   # скрытые слои
            'kan_width' : {'values' : [2 ** i for i in range(7)]},
            'grid_size' : {'values' : [i for i in range(3, 30, 2)]},
        })
    elif model_name == 'small_kan': # легкий KAN
        params.update({
            'kan_layers' : {'values' : [1, 2]},   # скрытые слои
            'kan_width' : {'values' : [4, 8, 12, 16, 20, 24]},
            'grid_size' : {'values' : [i for i in range(3, 13, 2)]},
        })
    elif model_name == 'cheby_kan': # Chebyshev-KAN
        params.update({
            'kan_layers' : {'values' : [1, 2, 3, 4]},   # скрытые слои
            'kan_width' : {'values' : [2 ** i for i in range(7)]},
            'degree' : {'values' : [i for i in range(1, 13)]} # попробуем такие степени
        })
    elif model_name == 'cheby_kan': # RBF-KAN
        params.update({
            'kan_layers' : {'values' : [1, 2, 3, 4]},   # скрытые слои
            'kan_width' : {'values' : [2 ** i for i in range(1, 7)]}, #
            'grid_size' : {'values' : [i for i in range(4, 30, 2)]} # пусть будут четные
        })
    elif model_name == 'kan_mlp' or model_name == 'mlp_kan':
        params.update({
            'kan_layers' : {'values' : [1, 2, 3]}, 
            'kan_width' : {'values' : [2 ** i for i in range(6)]},
            'grid_size' : {'values' : [i for i in range(3, 30, 2)]},
            'mlp_layers' : {'values' : [1, 2, 3]},
            'mlp_width' : {'values' : [2 ** i for i in range(10)]},
            'use_dropout' : {'values' : [True, False],
                             'probabilities' : [0.7, 0.3] # dropout вероятно нужен
                            },
            'dropout' : {'values' : [i / 100 for i in range(0, 55, 5)]}
        })

    if emb_name != 'none':
        params.update({
            'd_embedding' : {'values' : [2 ** i for i in range(1, 8)]}
        })
    if emb_name == 'periodic':
        params.update({
            'sigma' : {
                'distribution' : 'log_uniform_values',
                'min' : 0.01,
                'max' : 100
            }
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
        metric = {'name' : 'val_loss', 'goal' : 'minimize'}
    else:
        metric = {'name' : 'val_acc', 'goal' : 'maximize'}
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
 
