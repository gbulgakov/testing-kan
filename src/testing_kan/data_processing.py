import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

# Fixed batch sizes taken from TabularBench
BATCH_SIZES: Dict[str, int] = {
    "gesture": 128,
    "churn": 128,
    "california": 256,
    "house": 256,
    "adult": 256,
    "otto": 512,
    "higgs-small": 512,
    "fb-comments": 512,
    "microsoft": 1_024,
    "black-friday": 1_024,
    "diamond": 256,
}

class CustomTensorDataset(Dataset):
    """
    Simple tensor‐based dataset that always returns a triple
    (X_num, X_cat, y). If categorical inputs are absent, `X_cat`
    is replaced by a zero‐width tensor for easier downstream code.
    """

    def __init__(self, X_num: torch.Tensor, X_cat: torch.Tensor | None, y: torch.Tensor):
        self.X_num = X_num
        self.X_cat = X_cat if X_cat is not None else torch.zeros((len(X_num), 0))
        self.y = y

    def __len__(self) -> int:
        return len(self.X_num)

    def __getitem__(self, idx: int):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


class CustomBatchSampler(BatchSampler):
    """
    Custom batching logic that supports K-fold ensembling and
    “shared batches” training as used in KAN papers.
    """

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        k: int,
        share_batches: bool,
        device: str,
    ):
        super().__init__(None, batch_size, drop_last=False)
        self.data_size = data_size
        self.batch_size = batch_size
        self.k = k
        self.share_batches = share_batches
        self.device = device  # kept for interface parity

    
    def __iter__(self):
        # Shared batching → one permutation per epoch
        if self.share_batches:
            self.batches = torch.randperm(self.data_size, device="cpu").split(self.batch_size)
        # Otherwise each ensemble member draws its own permutation
        else:
            self.batches = [
                perm.transpose(0, 1).flatten()
                for perm in torch.rand((self.k, self.data_size), device="cpu")
                .argsort(dim=1)
                .split(self.batch_size, dim=1)
            ]
        self.num_batches = len(self.batches)
        return iter(self.batches)

    def __len__(self) -> int:
        return self.num_batches

    
def get_dataloader(
        model: nn.Module, 
        part: str, 
        batch_size: int, 
        dataset: Dataset, 
        device: str, 
        num_workers: int
) -> DataLoader:
    """
    Build a `DataLoader` with custom batching rules that match
    the parameter-efficient ensembles training pipeline.
    """
    
    # Share batches for every scenario *except* K-ensemble training
    share_batches = True
    if model.arch_type != "plain" and part == "train":
        share_batches = getattr(model, "share_training_batches", True)
    
    sampler = CustomBatchSampler(
        data_size=len(dataset),
        batch_size=batch_size,
        k=getattr(model, "k", 1),
        share_batches=share_batches,
        device=device,
    )

    return DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        pin_memory=(device != "cpu"),
        num_workers=num_workers,
    )
    

def get_dataloaders(
    dataset: Dict[str, Any],
    model,
    device: str,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """
    Wrap raw numpy tensors stored in `dataset` with `DataLoader` objects.
    """
    dataset_name = dataset['info']['id'].split('--')[0]
    loaders: Dict[str, DataLoader] = {}

    for part in ['train', 'test', 'val']:
        X_cat = dataset[part].get('X_cat', None)
        dataset_part = CustomTensorDataset(
            X_num=dataset[part]['X_num'],
            X_cat=X_cat,
            y=dataset[part]['y']
        )

        loaders[part] = get_dataloader(
            model=model,
            part=part,
            batch_size=BATCH_SIZES[dataset_name],
            dataset=dataset_part,
            device=device,
            num_workers=num_workers)
    return loaders


# ---------- I/O & preprocessing ------------------------------------------------


def load_dataset(name: str, zip_path: str | None = None) -> Dict[str, Any]:
    """
    Extract a pre-zipped dataset, apply simple preprocessing
    (standardisation + one-hot) and return a ready-to-use dict
    compatible with the rest of the pipeline.

    Notes
    -----
    * Numerical features are standardised with statistics from the
      training split only.
    * Labels for regression tasks are standardised.
    """
    data: Dict[str, Any] = {"train": {}, "val": {}, "test": {}}
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        temp_dir = Path(f'{name}_data')
        temp_dir.mkdir(exist_ok=True)
        zip_ref.extractall(temp_dir)

        content = list(temp_dir.glob("*"))
        data_dir = content[0] if len(content) == 1 and content[0].is_dir() else temp_dir / name
        data_dir.mkdir(exist_ok=True)

        for item in content:
            if item != data_dir:
                item.rename(data_dir / item.name)
    

        # Metadata
        with open(data_dir / "info.json") as f:
            data["info"] = json.load(f)

        # Encoders / scalers
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        # Load raw npy files
        for part in ['train', 'val', 'test']:
            for data_type in ['X_num', 'X_cat', 'y']:
                file_path = data_dir / f'{data_type}_{part}.npy'
                if file_path.exists():
                    data[part][data_type] = np.load(file_path, allow_pickle=True)
        
        # Categorical preprocessing (one-hot)
        cat_train = data["train"].get("X_cat")
        if cat_train is not None:
            ohe.fit(cat_train)
            for part in ("train", "val", "test"):
                if "X_cat" in data[part]:
                    data[part]["X_cat"] = ohe.transform(data[part]["X_cat"])

        # Numerical preprocessing (standardise)
        num_train = data["train"]["X_num"]
        scaler_X.fit(num_train)
        for part in ("train", "val", "test"):
            data[part]["X_num"] = scaler_X.transform(data[part]["X_num"])

        # Regression label scaling
        if data["info"]["task_type"] == "regression":
            y_train = data["train"]["y"].reshape(-1, 1)
            scaler_y.fit(y_train)
            for part in ("train", "val", "test"):
                y_part = data[part]["y"].reshape(-1, 1)
                data[part]["y"] = scaler_y.transform(y_part).reshape(-1)
            data["scaler_y"] = scaler_y  # for de-scaling later

        # Convert to tensors
        for part in ("train", "val", "test"):
            for key, arr in data[part].items():
                data[part][key] = torch.tensor(arr, dtype=torch.float)

        # Labels as `long` for multiclass classification  
        if data['info']['task_type'] == 'multiclass':
            for part in ['train', 'val', 'test']:
                data[part]['y'] = data[part]['y'].long()
            
        # Convenience fields
        info = data["info"]
        info["in_features"] = data["train"]["X_num"].shape[1]
        info["num_samples"] = sum(data[part]["X_num"].shape[0] for part in ("train", "val", "test"))
        info["num_cont_cols"] = data["train"]["X_num"].shape[1]
        info["num_cat_cols"] = 0

        if "X_cat" in data["train"]:
            info["in_features"] += data["train"]["X_cat"].shape[1]
            info["num_cat_cols"] = data["train"]["X_cat"].shape[1]
        
        # Clean up
        shutil.rmtree(temp_dir)
    return data
