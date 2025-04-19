from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR, StepLR, CosineAnnealingLR
from torch.nn import MSELoss
import torch.nn as nn
import time
import wandb
import utils
import torch
import delu


BATCH_SIZES = {'gesture' : 128, 'churn' : 128, 'california' : 256, 'house' : 256, 'adult' : 256, 'otto' : 512, 
               'higgs-small' : 512, 'fb-comments' : 512, 'santander' : 1024, 'covtype' : 1024, 'microsoft' : 1024, 'eye': 128}


def apply_model(batch: dict[str, Tensor], model) -> Tensor:
    return model(batch['X_num'], batch.get('X_cat')).squeeze(-1)

# одна эпоха
def train_epoch(model, device, dataset, loss_fn, optimizer, scheduler):

    dataset_name = dataset['info']['id'].split('--')[0]
    task_type = dataset['info']['task_type']
    batch_size = BATCH_SIZES[dataset_name]

    model.to(device)
    model.train()
    train_loss = 0.0
    pred = []
    gt = [] # настоящие таргеты
    start_time = time.time()

    for data in delu.iter_batches(dataset['train'], shuffle=True, batch_size=batch_size):
        for key, tensor in data.items():
            data[key] = tensor.to(device)
        # обучение
        optimizer.zero_grad()
        output = apply_model(data, model)
        if task_type == 'multiclass':
            data['y'] = data['y'].long()
        loss_value = loss_fn(output, data['y']) 
        loss_value.backward()
        optimizer.step()
        # сохранение истории
        train_loss += loss_value.item()
        if output.dim() > 1:
            pred.append(output.argmax(1))
        else:
            pred.append(output >= 0.5)
        gt.append(data['y'])
    scheduler.step()

    end_time = time.time()
    epoch_time = end_time - start_time

    num_batches = dataset['train']['y'].shape[0] // batch_size + 1
    pred = torch.cat(pred)
    gt = torch.cat(gt)
    train_accuracy = (pred == gt).float().mean().item()

    return train_loss / num_batches, train_accuracy, epoch_time # с нормировкой
    
# валидация
def validate(model, device, dataset, loss_fn, part='val'):
    model.eval()
    model.to(device)
    val_loss = 0.0

    pred = []
    gt = [] # настоящие таргеты

    dataset_name = dataset['info']['id'].split('--')[0]
    task_type = dataset['info']['task_type']
    batch_size = BATCH_SIZES[dataset_name]

    with torch.no_grad():
        start_time = time.time()
        for data in delu.iter_batches(dataset[part], shuffle=False, batch_size=batch_size):
            for key, tensor in data.items():
                data[key] = tensor.to(device)
            output = apply_model(data, model)
            if task_type == 'multiclass':
                data['y'] = data['y'].long()
            val_loss += loss_fn(output, data['y']).item()
            if output.dim() > 1:
                pred.append(output.argmax(1))
            else:
                pred.append(output >= 0.5)
            gt.append(data['y'])
        end_time = time.time()
        val_time = end_time - start_time
        

    num_batches = dataset[part]['y'].shape[0] // batch_size + 1
    pred = torch.cat(pred)
    gt = torch.cat(gt)
    val_accuracy = (pred == gt).float().mean().item()

    return val_loss / num_batches, val_accuracy, val_time # с нормировкой


#Возможно ``model.to()`` лучше вызывать 1 раз.
# обучение целиком
def train(
    epochs, model, model_name,
    device, dataset, loss_fn,
    optimizer
):
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)
    dataset_name = dataset['info']['id'].split('--')[0]
    task_type = dataset['info']['task_type']
    batch_size = BATCH_SIZES[dataset_name]

    train_times = []
    val_times = []
    for epoch in tqdm(range(epochs), desc = f'{model_name} on {dataset_name}'):
        train_loss, train_acc, train_time = train_epoch(model, device, dataset, loss_fn, optimizer, scheduler)
        val_loss, val_acc, val_time = validate(model, device, dataset, loss_fn)

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
        'num_params' : utils.count_parameters(model),
        'in_features' : in_features,
        'out_features' : (1 if task_type != 'multiclass' else dataset['info']['n_classes'])
        # ширины и так будут залоггированы
    })