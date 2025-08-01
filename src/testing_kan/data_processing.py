import shutil
import zipfile
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
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
    'homesite-insurance' : 512,
    'homecredit-default' : 512,
    'ecom-offers' : 512, 
    'regression-num-large-0-year' : 1024,
    'black-friday' : 1024,
    'diamond' : 256,
    'classif-num-large-0-MiniBooNE' : 512,
    'regression-num-medium-0-medical_charges' : 1024,
    'classif-cat-large-0-road-safety' : 1024,
    'regression-cat-large-0-nyc-taxi-green-dec-2016' : 1024,
    'regression-cat-large-0-particulate-matter-ukair-2017' : 1024
}

# dataset with proper batches
class CustomTensorDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = X_num
        self.X_cat = X_cat if X_cat is not None else torch.zeros((len(X_num), 0))
        self.y = y

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

# custom batch sampling
class CustomBatchSampler(BatchSampler):
    def __init__(self, data_size, batch_size, k, share_batches, device):
        self.data_size = data_size
        self.batch_size = batch_size
        self.k = k
        self.share_batches = share_batches
        self.device = device
    
    def __iter__(self):
        # iter on generated batches
        if self.share_batches:
            # one permotutation - common case
            self.batches = torch.randperm(self.data_size, device='cpu').split(self.batch_size) 
        else:
            self.batches = [
                x.transpose(0, 1).flatten()
                for x in torch.rand((self.k, self.data_size), device='cpu')
                .argsort(dim=1)
                .split(self.batch_size, dim=1)
            ]
        # num batches
        self.num_batches = len(self.batches)
        return iter(self.batches)
    
    def __len__(self):
        return self.num_batches
    
def get_dataloader(model, part: str, batch_size: int, dataset: Dataset, device: str, num_workers: int) -> DataLoader:
    arch_type = model.arch_type
    data_size = len(dataset)
    k = getattr(model, 'k', 1)
    
    # batching strategy
    if arch_type != 'plain' and part == 'train':
        share_batches = getattr(model, 'share_training_batches', True)
    else:
        # other cases (eval/test) -> share batches
        share_batches = True
    
    batch_sampler = CustomBatchSampler(
        data_size=data_size,
        batch_size=batch_size,
        k=k,
        share_batches=share_batches,
        device=device,
    )
    
    # dataloader with our batch sampler
    return DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        pin_memory=(device != 'cpu'),
        num_workers=num_workers
        # collate_fn=collate_fn
    )

def get_dataloaders(dataset, model, device, num_workers=4):
    dataset_name = dataset['info']['id'].split('--')[0]
    dataloaders = {}
    for part in ['train', 'test', 'val']:
        X_cat = dataset[part].get('X_cat', None)
        dataset_part = CustomTensorDataset(
            X_num=dataset[part]['X_num'],
            X_cat=X_cat,
            y=dataset[part]['y']
        )

        dataloaders[part] = get_dataloader(
            model=model,
            part=part,
            batch_size=BATCH_SIZES[dataset_name],
            dataset=dataset_part,
            device=device,
            num_workers=num_workers)
    return dataloaders

# simple preprocessing
def load_dataset(name, zip_path=None):
    if zip_path is None:
        zip_path = f'/kaggle/working/{name}.zip'
    data = {'train': {}, 'val': {}, 'test': {}}
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        temp_dir = Path(f'{name}_data')
        temp_dir.mkdir(exist_ok=True)
        zip_ref.extractall(temp_dir)

        content = list(temp_dir.glob('*'))
        if len(content) == 1 and content[0].is_dir():
            data_dir = content[0]
        else:
            data_dir = temp_dir / name
            data_dir.mkdir(exist_ok=True)
            for item in content:
                if item != data_dir:
                    item.rename(data_dir / item.name)
    

        with open(data_dir / 'info.json') as f:
            data['info'] = json.load(f)
        

        # cat features
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # num features and labels (regression)
        scaler = StandardScaler()
        scaler_y = StandardScaler()
        
        for part in ['train', 'val', 'test']:
            for data_type in ['X_num', 'X_cat', 'y']:
                file_path = data_dir / f'{data_type}_{part}.npy'
                if file_path.exists():
                    data[part][data_type] = np.load(file_path, allow_pickle=True)

        # cat features
        for part in ['train', 'val', 'test']:
            cat_path = data_dir / f'X_cat_{part}.npy'
            if cat_path.exists():
                if part == 'train':
                    one_hot_encoder.fit(data['train']['X_cat']) 
                data[part]['X_cat'] = one_hot_encoder.transform(data[part]['X_cat'])

        # StandardScaler on train 
        for part in ['train', 'val', 'test']:
            num_path = data_dir / f'X_num_{part}.npy'
            if num_path.exists():
                if part == 'train':
                    scaler.fit(data['train']['X_num'])  # Обучаем scaler только на train
                data[part]['X_num'] = scaler.transform(data[part]['X_num'])

        # y standatization for regressions
        if data['info']['task_type'] == 'regression':
            for part in ['train', 'val', 'test']:
                y_path = data_dir / f'y_{part}.npy'
                if y_path.exists():
                    if part == 'train':
                        # reshape to 2D
                        y_reshaped = data['train']['y'].reshape(-1, 1)
                        scaler_y.fit(y_reshaped)
                    
                    # reshape 
                    y_reshaped = data[part]['y'].reshape(-1, 1)
                    data[part]['y'] = scaler_y.transform(y_reshaped).reshape(data[part]['y'].shape)
            data['scaler_y'] = scaler_y

        # back to torch.tensor
        for part in ['train', 'val', 'test']:
            for data_type in data[part].keys():
                data[part][data_type] = torch.tensor(data[part][data_type], dtype=torch.float)

        # convert to long type for multiclass    
        if data['info']['task_type'] == 'multiclass':
            for part in ['train', 'val', 'test']:
                data[part]['y'] = data[part]['y'].long()
            
        # for convinience
        data['info']['in_features'] = data['train']['X_num'].shape[1]
        data['info']['num_samples'] = 0
        for part in ['train', 'test', 'val']:
            data['info']['num_samples'] += data[part]['X_num'].shape[0]
        data['info']['num_cont_cols'] = data['train']['X_num'].shape[1]
        data['info']['num_cat_cols'] = 0
        if 'X_cat' in data['train']:
            data['info']['in_features'] += data['train']['X_cat'].shape[1]
            data['info']['num_cat_cols'] = data['train']['X_cat'].shape[1]
        
        shutil.rmtree(temp_dir)
    
    return data
