import time
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
# import delu пока 

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

def get_batches_indices(model, arch_type: str, part: str, batch_size: int, data_size, device) -> torch.Tensor:
    if arch_type != 'plain' and part == 'train':
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
            y_true.repeat_interleave(y_pred.shape[-1 if task_type != 'multiclass' else -2]) if share_training_batches else y_true,
        )
    else:
        loss_fn = lambda y_pred, y_true: base_loss_fn(y_pred.squeeze(1), y_true)
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
def apply_model(batch_data: dict, model) -> torch.Tensor:
    '''Сюда поступает батч на нужном девайсе и на нужной части датасета'''
    return (
        model(
            batch_data['X_num'],
            batch_data.get('X_cat')
        )
        .squeeze(-1)  # Remove the last dimension for regression tasks.
        .float()
    )

def train_epoch(model, device, dataset, base_loss_fn, optimizer, scheduler, model_name, arch_type):
    for key, tensor in dataset['train'].items():  # наши датасеты спокойно влезают в память
        dataset['train'][key] = tensor.to(device) # overkill, скорее всего нужно сделать умнее
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
    batches = get_batches_indices(model, arch_type, 'train', batch_size, train_size, device=device)

    for batch_indices in batches:
        # для удобства
        batch_data = {
            key: dataset['train'][key][batch_indices]
            for key in dataset['train'].keys()
        }

        optimizer.zero_grad()
        
        # (B, k, n_out) or (B, k) (if regression)
        output = apply_model(batch_data, model)
        loss_value = loss_fn(output, batch_data['y'])
        loss_value.backward()
        optimizer.step()
        # сохранение истории
        train_loss += loss_value.item()
        
        #not needed for regression
        if model.share_training_batches:
            output = output.mean(dim=1) # (B, n_out) or (B)
            y_true = batch_data['y'] # (B)
        else:
            output = output # (B, k, n_out) or (B, k)
            y_true = batch_data['y'] # (B, k)
        
        
        n_bound = 1 if model.share_training_batches else 2
        if output.dim() > n_bound:
            pred.append(output.argmax(1)) # if multiclass then -> argmax over classes (pred (B, k) or (B))
        else:
            pred.append(output >= 0.5) # if binaryclassification then -> >= 0.5 (pred (B, k) or (B))
        gt.append(y_true)
    scheduler.step()

    end_time = time.time()
    epoch_time = end_time - start_time

    num_batches = dataset['train']['y'].shape[0] // batch_size + 1
    pred = torch.cat(pred)
    gt = torch.cat(gt) #(dataset_size, k) or (dataset_size)
    #accuracy is found as accuracy of mean prediction over k if model.share_training_batches==True
    # else accuracy of all predictions
    train_accuracy = (pred == gt).float().mean().item()

    return train_loss / num_batches, train_accuracy, epoch_time # с нормировкой
    
def validate(model, device, dataset, base_loss_fn, part, model_name: str, arch_type):
    for key, tensor in dataset[part].items():
        dataset[part][key] = tensor.to(device) #overkill, скорее всего нужно сделать умнее
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

    # мб стоит оптимизировать
    if task_type == 'multiclass':
        dataset[part]['y'] = dataset[part]['y'].long()
    
    with torch.no_grad():
        start_time = time.time()
        for batch_indices in batches:
            # для удобства
            batch_data = {
                key: dataset[part][key][batch_indices]
                for key in dataset[part].keys()
            }

            output = apply_model(batch_data, model) # (B, k, n_out) or (B, k)

            val_loss += loss_fn(output, batch_data['y']).item()
            
            output = output.mean(dim=1) # (B, n_out) or (B)
            
            if output.dim() > 1:
                pred.append(output.argmax(1)) 
                #output.argmax(1) -> (B)
            else:
                pred.append(output >= 0.5)
                #output >= 0.5 -> (B)
            gt.append(batch_data['y'])
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

    # приведение к long можно сделать 1 раз, а не на каждой эпохе
    if task_type == 'multiclass':
        dataset['train']['y'] = dataset['train']['y'].long()
        

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
