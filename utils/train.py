import time
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
# import delu пока убрал из использования

from utils.utils import count_parameters

# Словарь размеров батчей для разных датасетов
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
    'eye': 128
}


def select_batch(data: dict, batch_indices: torch.Tensor) -> dict: # костыльная функция для того, чтобы переводить индексы батча в привычный формат data
    selected_data = {}
    
    # Обрабатываем вложенные словари с тензорами
    for key, value in data.items():
        if isinstance(value, dict):
            selected_data[key] = {
                sub_key: sub_value[batch_indices] 
                for sub_key, sub_value in value.items()
            }
        elif isinstance(value, torch.Tensor):
            selected_data[key] = value[batch_indices]
        else:
            selected_data[key] = value  # если не тензор, оставляем как есть
    
    return selected_data

def get_batches_indices(model, arch_type: str, part: str, batch_size: int, data_size, device) -> torch.Tensor:
    if arch_type != 'plain':
        batches = (
        torch.randperm(data_size, device=device).split(batch_size)
        if model.share_training_batches
        else [
            x.transpose(0, 1).flatten()
            for x in torch.rand((model.k, data_size), device=device)
            .argsort(dim=1)
            .split(batch_size, dim=1)
        ]
        )
    else:
        batches = torch.randperm(data_size, device=device).split(batch_size)
    return batches


def get_loss_fn(arch_type: str, base_loss_fn: str, task_type: str, share_training_batches: bool):
    if arch_type != 'plain':
        loss_fn = lambda y_pred, y_true: base_loss_fn(
            y_pred.flatten(0, 1),
            y_true.repeat_interleave(y_pred.shape[-1 if task_type == 'regression' else -2]) if share_training_batches else y_true,
        )
    else:
        loss_fn = base_loss_fn
    return loss_fn


# def apply_model(batch: dict[str, torch.Tensor], model) -> torch.Tensor:
#     return model(batch['X_num'], batch.get('X_cat')).squeeze(-1)
# Automatic mixed precision (AMP)
# torch.float16 is implemented for completeness,
# but it was not tested in the project,
# so torch.bfloat16 is used by default.
amp_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
    if torch.cuda.is_available()
    else None
)
# Changing False to True will result in faster training on compatible hardware.
amp_enabled = False and amp_dtype is not None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
def apply_model(part: str, idx: torch.Tensor, data: dict, model) -> torch.Tensor:
    return (
        model(
            data[part]['X_num'][idx],
            data[part]['X_cat'][idx] if 'X_cat' in data[part] else None,
        )
        .squeeze(-1)  # Remove the last dimension for regression tasks.
        .float()
    )

def train_epoch(model, device, dataset, base_loss_fn, optimizer, scheduler, model_name, arch_type):
    for key, tensor in dataset['train'].items():
        tensor.to(device) #overkill, скорее всего нужно сделать умнее
    dataset_name = dataset['info']['id'].split('--')[0]
    task_type = dataset['info']['task_type']
    batch_size = BATCH_SIZES[dataset_name]
    train_size = dataset['train']['X_num'].shape[0]

    model.to(device)
    model.train()
    train_loss = 0.0
    pred = []
    gt = [] # настоящие таргеты
    start_time = time.time()
    
    loss_fn = get_loss_fn(arch_type, base_loss_fn, task_type, model.share_training_batches)
    batches = get_batches_indices(model, arch_type, 'train', batch_size, train_size, device)
        
    for batch_indices in batches:
        # for key, tensor in data.items():
        #     data[key] = tensor.to(device)
        # обучение
        optimizer.zero_grad()
        output = apply_model('train', batch_indices, dataset, model)
        if task_type == 'multiclass':
            dataset['train']['y'] = dataset['train']['y'].long()
        loss_value = loss_fn(output, dataset['train']['y']) 
        loss_value.backward()
        optimizer.step()
        # сохранение истории
        train_loss += loss_value.item()
        if output.dim() > 1:
            pred.append(output.argmax(1))
        else:
            pred.append(output >= 0.5)
        gt.append(dataset['train']['y'])
    scheduler.step()

    end_time = time.time()
    epoch_time = end_time - start_time

    num_batches = dataset['train']['y'].shape[0] // batch_size + 1
    pred = torch.cat(pred)
    gt = torch.cat(gt)
    train_accuracy = (pred == gt).float().mean().item()

    return train_loss / num_batches, train_accuracy, epoch_time # с нормировкой
    
def validate(model, device, dataset, base_loss_fn, part, model_name: str, arch_type):
    for key, tensor in dataset[part].items():
        tensor.to(device) #overkill, скорее всего нужно сделать умнее
    model.eval()
    model.to(device)
    val_loss = 0.0
    val_size = dataset['val']['X_num'].shape[0]

    pred = []
    gt = [] # настоящие таргеты

    dataset_name = dataset['info']['id'].split('--')[0]
    task_type = dataset['info']['task_type']
    batch_size = BATCH_SIZES[dataset_name]
    
    batches = get_batches_indices(model, model_name, part, batch_size, val_size, device)
    loss_fn = get_loss_fn(model_name, base_loss_fn, task_type, model.share_training_batches)
    
    with torch.no_grad():
        start_time = time.time()
        for batch_indices in batches:
            # for key, tensor in data.items():
            #     data[key] = tensor.to(device)
            output = apply_model(part, batch_indices, dataset, model)
            if task_type == 'multiclass':
                dataset[part]['y'] = dataset[part]['y'].long()
            val_loss += loss_fn(output, dataset[part]['y']).item()
            if output.dim() > 1:
                pred.append(output.argmax(1))
            else:
                pred.append(output >= 0.5)
            gt.append(dataset[part]['y'])
        end_time = time.time()
        val_time = end_time - start_time
        
    num_batches = dataset[part]['y'].shape[0] // batch_size + 1
    pred = torch.cat(pred)
    gt = torch.cat(gt)
    val_accuracy = (pred == gt).float().mean().item()

    return val_loss / num_batches, val_accuracy, val_time # с нормировкой

def train(
    epochs, model, model_name, arch_type,
    device, dataset, base_loss_fn,  
    optimizer
):
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)
    dataset_name = dataset['info']['id'].split('--')[0]
    task_type = dataset['info']['task_type']
    batch_size = BATCH_SIZES[dataset_name]

    train_times = []
    val_times = []
    for epoch in tqdm(range(epochs), desc = f'{model_name}_{arch_type} on {dataset_name}'):
        train_loss, train_acc, train_time = train_epoch(model, device, dataset, base_loss_fn, optimizer, scheduler, model_name, arch_type)
        val_loss, val_acc, val_time = validate(model, device, dataset, base_loss_fn, 'val', model_name, arch_type)

        wandb.log({
            'epoch' : epoch,
            'train_loss' : train_loss,
            'train_acc' : train_acc,
            'val_loss' : val_loss,
            'val_acc' : val_acc,
            'lr' : scheduler.get_last_lr()[0]
        })
        train_times.append(train_time)
        val_times.append(val_time)

    # размерность входа backbone
    in_features = dataset['train']['X_num'].shape[1]  # Количество числовых признаков
    if 'X_cat' in dataset['train']:
        in_features += dataset['train']['X_cat'].shape[1]  # Добавляем категориальные признаки

    
    wandb.log({
        'train_epoch_time' : sum(train_times) / epochs,
        'val_epoch_time' : sum(val_times) / epochs,
        'num_params' : count_parameters(model),
        'in_features' : in_features,
        'out_features' : (1 if task_type != 'multiclass' else dataset['info']['n_classes'])
        # ширины и так будут залоггированы
    })
