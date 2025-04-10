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

    # !cd /content/experiments_with_kan && git config --global user.email "ваш-email" && git config --global user.name "ваш-username"
    # !cd /content/experiments_with_kan && git add . && git commit -m "Добавлен файл модели" && git push