from copy import deepcopy
from torch import mode
import typing as ty
import numpy as np
import sklearn.metrics as skm
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import shutil
import zipfile
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import torch

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

def get_sweep_config(model_name, task_type, sweep_name):
    if task_type == 'regression':
        metric = {'name' : 'val_loss', 'goal' : 'minimize'}
    else:
        metric = {'name' : 'val_accuracy', 'goal' : 'maximize'}
    
    params = {
        'learning_rate' : {
            'distribution' : 'log_uniform_values',
            'min' : 1e-3,
            'max' : 1 # 
        }
    }
    if model_name == 'catboost':
        params.update({
            'bagging_temperature' : {'distribution': 'uniform',
                              'min': 0,
                              'max': 1},
            'depth' : {'values': [i for i in range(3, 11)]},
            'l2_leaf_reg' : {'distribution': 'uniform',
                              'min': 1e-1,
                              'max': 10},
            'leaf_estimation_iterations' : {'values': [i for i in range(1, 11)]},
        })
    
    if model_name == 'xgboost':
        params.update({
            'colsample_bytree': {'distribution': 'uniform',
                              'min': 1/2,
                              'max': 1},
            'colsample_bylevel': {'distribution': 'uniform',
                              'min': 1/2,
                              'max': 1},
            'gamma': {'distribution': 'uniform',
                              'min': 1e-3,
                              'max': 100},
            'lambda': {'distribution': 'log_uniform',
                              'min': 1e-1,
                              'max': 10},
            'max_depth': {'values': [i for i in range(3, 11)]},
            'min_child_weight': {'distribution': 'log_uniform',
                              'min': 1e-8,
                              'max': 1e5},
            'subsample': {'distribution': 'uniform',
                              'min': 1/2,
                              'max': 1}
            })
    if model_name == 'lightgbm':
        params.update({
            'num_leaves': {'values': [i for i in range(10, 101)]},
            'min_child_weight': {'distribution': 'log_uniform',
                              'min': 1e-5,
                              'max': 1e-1},
            'min_child_samples' : {'values': [i for i in range(2, 101)]},
            'subsample': {'distribution': 'uniform',
                              'min': 1/2,
                              'max': 1},
            'colsample_bytree': {'distribution': 'uniform',
                              'min': 1/2,
                              'max': 1},
            'reg_lambda': {'distribution': 'log_uniform',
                           'min': 1e-5,
                           'max': 1}
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
        metric = {'name' : 'val_accuracy', 'goal' : 'maximize'}
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

def merge_sampled_parameters(config, sampled_parameters):
    for k, v in sampled_parameters.items():
        if isinstance(v, dict):
            merge_sampled_parameters(config.setdefault(k, {}), v)
        else:
            assert k not in config
            config[k] = v

def suggest_params(trial, model_name, task_type):
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1),
    }
    
    if model_name == 'catboost':
        params.update({
            'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0, 1),
            'depth': trial.suggest_int('depth', 3, 10),  # Аналог 'values' [3..10]
            'l2_leaf_reg': trial.suggest_uniform('l2_leaf_reg', 1e-1, 10),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),  # Аналог 'values' [1..10]
        })
    
    if model_name == 'xgboost':
        params.update({
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1),
            'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.5, 1),
            'gamma': trial.suggest_uniform('gamma', 1e-3, 100),
            'lambda': trial.suggest_loguniform('lambda', 0.1, 10),
            'max_depth': trial.suggest_int('max_depth', 3, 10),  # Аналог 'values' [3..10]
            'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-8, 1e5),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1),
        })
    
    if model_name == 'lightgbm':
        params.update({
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),  # Аналог 'values' [10..100]
            'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-5, 0.1),
            'min_child_samples': trial.suggest_int('min_child_samples', 2, 100),  # Аналог 'values' [2..100]
            'subsample': trial.suggest_uniform('subsample', 0.5, 1),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 1),
        })
    
    # Если нужно добавить параметры, зависящие от task_type (например, для регрессии/классификации), сделайте это здесь
    # Пример: if task_type == 'regression': params['some_param'] = ...
    
    return params

def model_init_preparation(config, dataset, model_name):
    #here config is already sampled with wandb

    # print(type(config), config)

    if 'parameters' in config.keys():
        model_config = {key: config[key] for key in config['parameters'].keys()}
    else:
        model_config = config

    dataset_info = dataset['info']
    task_type = dataset_info['task_type']
    num_cont_cols = dataset_info['num_cont_cols']
    num_cat_cols = dataset_info['num_cat_cols']
    num_classes = 1
    if task_type == 'multiclass':
        num_classes = dataset_info['n_classes']
    if model_name == 'catboost':
        model_kwargs = {
            'early_stopping_rounds': 50,
            'iterations': 2000,
            'metric_period': 10,
            'od_pval': 0.001,
            'thread_count': 1,
            'task_type' : 'GPU'
        }
        merge_sampled_parameters(model_kwargs, model_config)
        fit_kwargs = {'logging_level': 'Verbose',
                      'eval_set': (dataset['val']['X'], dataset['val']['y'])}
        if task_type == 'regression':
            model = CatBoostRegressor(**model_kwargs)
            predict = model.predict
        else:
            model = CatBoostClassifier(**model_kwargs, eval_metric='Accuracy')
            predict = (
                model.predict_proba
                if task_type == 'multiclass'
                else lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]
            )
    if model_name == 'xgboost':
        model_kwargs = {
            'booster': 'gbtree',
            'n_estimators': 2000,
            'n_jobs': -1,
            'tree_method': 'hist',
            'device': 'cuda',
            'early_stopping_rounds': 50,
        }
        merge_sampled_parameters(model_kwargs, model_config)
        fit_kwargs = {'verbose': False,
                      'eval_set': [(dataset['val']['X'], dataset['val']['y'])]}
        if task_type == 'regression':
            model = XGBRegressor(**model_kwargs)
            predict = model.predict
        else:
            if task_type == 'multiclass':
                model_kwargs['eval_metric'] = 'merror'
                model = XGBClassifier(**model_kwargs, disable_default_eval_metric=True)
                predict = model.predict_proba
            else:
                model_kwargs['eval_metric'] = 'error'
                model = XGBClassifier(**model_kwargs, disable_default_eval_metric=True)
                predict = lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]  # noqa
    if model_name == 'lightgbm':
        model_kwargs = {
            'n_estimators': 2000,
            'n_jobs': 1,
            'device_type': 'gpu'
        }
        merge_sampled_parameters(model_kwargs, model_config)
        fit_kwargs = {'eval_set': (dataset['val']['X'], dataset['val']['y'])}
        if task_type == 'regression':
            model = LGBMRegressor(**model_kwargs)
            fit_kwargs['eval_metric'] = 'rmse'
            predict = model.predict
        else:
            model = LGBMClassifier(**model_kwargs)
            if task_type == 'multiclass':
                predict = model.predict_proba
                fit_kwargs['eval_metric'] = 'multi_error'
            else:
                predict = lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]  # noqa
                fit_kwargs['eval_metric'] = 'binary_error'
    return model, fit_kwargs, predict

# добавил простейший препроцессинг
def load_dataset(name, zip_path=None):
    if zip_path is None:
        zip_path = f'/kaggle/working/{name}.zip'
    data = {'train': {}, 'val': {}, 'test': {}}
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Создаем временную директорию для распаковки
        temp_dir = Path(f'{name}_data')
        temp_dir.mkdir(exist_ok=True)

        for file in zip_ref.namelist():
            if not file.startswith('__MACOSX/'):
                zip_ref.extract(file, temp_dir)

        # Определяем корневую папку с данными
        content = [item for item in temp_dir.glob('*') if not item.name.startswith('__MACOSX')]
        if len(content) == 1 and content[0].is_dir():
            # Если в архиве была одна папка - используем её
            data_dir = content[0]
        else:
            # Если файлы были в корне - создаем папку с именем датасета
            data_dir = temp_dir / name
            data_dir.mkdir(exist_ok=True)
            for item in content:
                if item != data_dir:
                    try:
                        item.rename(data_dir / item.name)
                    except OSError as e:
                        # Если не удалось переместить из-за непустой директории
                        shutil.move(str(item), str(data_dir / item.name))
    

        # Загружаем метаданные
        with open(data_dir / 'info.json') as f:
            data['info'] = json.load(f)
        

        # Для категориальных фич
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Для числовых фич и меток (регрессия)
        scaler = StandardScaler()
        scaler_y = StandardScaler()
        
        # Загружаем все .npy файлы
        for part in ['train', 'val', 'test']:
            for data_type in ['X_num', 'X_cat', 'y']:
                file_path = data_dir / f'{data_type}_{part}.npy'
                if file_path.exists():
                    data[part][data_type] = np.load(file_path, allow_pickle=True)

        # Обучение OHE на train и кодирование категориальных признаков
        for part in ['train', 'val', 'test']:
            cat_path = data_dir / f'X_cat_{part}.npy'
            if cat_path.exists():
                if part == 'train':
                    one_hot_encoder.fit(data['train']['X_cat']) 
                data[part]['X_cat'] = one_hot_encoder.transform(data[part]['X_cat'])

        # Обучение StandardScaler на train и стандартизация числовых признаков
        for part in ['train', 'val', 'test']:
            num_path = data_dir / f'X_num_{part}.npy'
            if num_path.exists():
                if part == 'train':
                    scaler.fit(data['train']['X_num'])  # Обучаем scaler только на train
                data[part]['X_num'] = scaler.transform(data[part]['X_num'])

        # Для регрессии надо стандартизировать y
        if data['info']['task_type'] == 'regression':
            for part in ['train', 'val', 'test']:
                y_path = data_dir / f'y_{part}.npy'
                if y_path.exists():
                    if part == 'train':
                        # Преобразуем в 2D для StandardScaler, сохраняя размерность
                        y_reshaped = data['train']['y'].reshape(-1, 1)
                        scaler_y.fit(y_reshaped)
                    
                    # Преобразуем, сохраняя оригинальную размерность
                    y_reshaped = data[part]['y'].reshape(-1, 1)
                    data[part]['y'] = scaler_y.transform(y_reshaped).reshape(data[part]['y'].shape)
            data['scaler_y'] = scaler_y

        # добавим полезную инфу
        data['info']['in_features'] = data['train']['X_num'].shape[1]
        data['info']['num_samples'] = 0
        for part in ['train', 'test', 'val']:
            data['info']['num_samples'] += data[part]['X_num'].shape[0]
        data['info']['num_cont_cols'] = data['train']['X_num'].shape[1]
        data['info']['num_cat_cols'] = 0
        if 'X_cat' in data['train']:
            data['info']['in_features'] += data['train']['X_cat'].shape[1]
            data['info']['num_cat_cols'] = data['train']['X_cat'].shape[1]

        # Переводим данные в тензоры'

        for part in ['train', 'val', 'test']:
            if 'X_cat' in data[part]:
                data[part]['X'] = np.hstack([data[part]['X_num'], data[part]['X_cat']])
            else:
                data[part]['X'] = data[part]['X_num']
            data[part].pop('X_num', None)
            data[part].pop('X_cat', None)
            

        # for part in ['train', 'val', 'test']:
        #     for data_type in data[part].keys():
        #         data[part][data_type] = torch.tensor(data[part][data_type], dtype=torch.float)

        # для multiclass нужно привести все к long        
        # if data['info']['task_type'] == 'multiclass':
        #     for part in ['train', 'val', 'test']:
        #         data[part]['y'] = data[part]['y'].long()
        
        # Удаляем временную директорию
        shutil.rmtree(temp_dir)
    
    # dataset = get_dataloaders(data, num_workers=num_workers)
    # dataset['info'] = data['info']
    # return dataset
    #
    #Перенес get_dataloaders вне datasets.py т.к. теперь его применение требует знания k, arch_type
    
    return data


