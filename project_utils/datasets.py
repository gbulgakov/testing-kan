import shutil
import zipfile
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import pickle
import json
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
BATCH_SIZES = {
    'gesture': 128,
    'churn': 128,
    'california': 256,
    'house': 256,
    'adult': 256,
    'otto': 512,
    'higgs-small': 512,
    'fb-comments': 512,
    'santander': 1024,
    'covtype': 1024,
    'microsoft': 1024,
    'eye': 128,
    'sberbank-housing' : 256,
    'homesite-insurance' : 1024,
    'homecredit-default' : 1024,
    'ecom-offers' : 1024, 
    'regression-num-large-0-year' : 1024,
    'black-friday' : 1024
}

# датасет с батчами нужного вида
class CustomTensorDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = X_num
        self.X_cat = X_cat if X_cat is not None else torch.zeros((len(X_num), 0))
        self.y = y

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

# это для получения даталоадеров потом 
def get_dataloaders(dataset, num_workers=4):
    dataset_name = dataset['info']['id'].split('--')[0]
    dataloaders = {}
    for part in ['train', 'test', 'val']:
        # проверяем наличие X_cat
        X_cat = dataset[part].get('X_cat', None)
        dataset_part = CustomTensorDataset(
            X_num=dataset[part]['X_num'],
            X_cat=X_cat,
            y=dataset[part]['y']
        )

        dataloaders[part] = DataLoader(
            dataset=dataset_part,
            batch_size=BATCH_SIZES[dataset_name],
            shuffle=(part == 'train'),
            num_workers=num_workers,
            pin_memory=True         # потому что всегда cuda
        )
    return dataloaders

# добавил простейший препроцессинг
def load_dataset(name, zip_path=None, num_workers=4):
    if zip_path is None:
        zip_path = f'/kaggle/working/{name}.zip'
    data = {'train': {}, 'val': {}, 'test': {}}
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Создаем временную директорию для распаковки
        temp_dir = Path(f'{name}_data')
        temp_dir.mkdir(exist_ok=True)
        zip_ref.extractall(temp_dir)

        # Определяем корневую папку с данными
        content = list(temp_dir.glob('*'))
        if len(content) == 1 and content[0].is_dir():
            # Если в архиве была одна папка - используем её
            data_dir = content[0]
        else:
            # Если файлы были в корне - создаем папку с именем датасета
            data_dir = temp_dir / name
            data_dir.mkdir(exist_ok=True)
            for item in content:
                item.rename(data_dir / item.name)
    

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

        # Переводим данные в тензоры
        for part in ['train', 'val', 'test']:
            for data_type in data[part].keys():
                data[part][data_type] = torch.tensor(data[part][data_type], dtype=torch.float)

        # для multiclass нужно привести все к long        
        if data['info']['task_type'] == 'multiclass':
            for part in ['train', 'val', 'test']:
                data[part]['y'] = data[part]['y'].long()
            
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
        
        # Удаляем временную директорию
        shutil.rmtree(temp_dir)
    
    dataset = get_dataloaders(data, num_workers=num_workers)
    dataset['info'] = data['info']
    return dataset
