import time

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.testing_kan.utils import count_parameters, compare_epochs

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
    'eye': 128,
    'sberbank-housing' : 256,
    'homesite-insurance' : 1024,
    'homecredit-default' : 1024,
    'ecom-offers' : 1024, 
    'regression-num-large-0-year' : 1024,
    'black-friday' : 1024,
    'diamond' : 256,
    'classif-num-large-0-MiniBooNE' : 512,
    'regression-num-medium-0-medical_charges' : 1024,
    'classif-cat-large-0-road-safety' : 1024,
    'regression-cat-large-0-nyc-taxi-green-dec-2016' : 1024,
    'regression-cat-large-0-particulate-matter-ukair-2017' : 1024
}

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
device = torch.device('cuda')
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
    # for key, tensor in dataset['train'].items():  # наши датасеты спокойно влезают в память
    #     dataset['train'][key] = tensor.to(device) # overkill, скорее всего нужно сделать умнее
    dataset_name = dataset['info']['id'].split('--')[0]
    task_type = dataset['info']['task_type']
    loader = dataset['train']
    # train_size = dataset['train']['X_num'].shape[0]

    model.to(device)
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    loss_fn = get_loss_fn(arch_type, base_loss_fn, task_type, model.share_training_batches)

    for (X_num, X_cat, y) in loader:
        X_num = X_num.to(device)
        X_cat = X_cat.to(device)
        y = y.to(device)
        batch_data = {
            'X_num' : X_num, 
            'X_cat' : X_cat if X_cat.size(1) > 0 else None, 
            'y' : y
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
        
        
        if task_type == 'binclass' or task_type == 'regression':
            pred = (output >= 0).float() # if binaryclassification then -> >= 0.5 (pred (B, k) or (B))
        else:
            pred = output.argmax(1) # if multiclass then -> argmax over classes (pred (B, k) or (B))
        
        correct += (pred == y_true).float().sum().item()
        total += y_true.numel()
    scheduler.step()

    end_time = time.time()
    epoch_time = end_time - start_time

    num_batches = len(loader)
    train_accuracy = correct / total
    #accuracy is found as accuracy of mean prediction over k if model.share_training_batches==True
    # else accuracy of all predictions

    return train_loss / num_batches, train_accuracy, epoch_time # с нормировкой
    
def validate(model, device, dataset, base_loss_fn, part, model_name: str, arch_type):
    model.eval()
    model.to(device)
    val_loss = 0.0
    loader = dataset[part]
    # val_size = dataset['val']['X_num'].shape[0]

    correct = 0
    total = 0

    task_type = dataset['info']['task_type']
    
    loss_fn = get_loss_fn(model_name, base_loss_fn, task_type, model.share_training_batches)
    
    with torch.no_grad():
        start_time = time.time()

        for (X_num, X_cat, y) in loader:
            X_num = X_num.to(device)
            X_cat = X_cat.to(device)
            y = y.to(device)
            batch_data = {
                'X_num' : X_num, 
                'X_cat' : X_cat if X_cat.size(1) > 0 else None, 
                'y' : y
            }

            output = apply_model(batch_data, model) # (B, k, n_out) or (B, k)

            val_loss += loss_fn(output, batch_data['y']).item()
            
            output = output.mean(dim=1) # (B, n_out) or (B)

            #not needed for regression
            if task_type == 'binclass' or task_type == 'regression':
                pred = (output >= 0).float()
                #output >= 0.5 -> (B)
            else:
                pred = output.argmax(1)  #multiclass
                #output.argmax(1) -> (B)
                        
            correct += (pred == batch_data['y']).float().sum().item()
            total += batch_data['y'].numel()

        end_time = time.time()
        val_time = end_time - start_time
        
    # num_batches = dataset[part]['y'].shape[0] // batch_size + 1
    num_batches = len(loader)
    val_accuracy = correct / total

    return val_loss / num_batches, val_accuracy, val_time # с нормировкой

def train(
    epochs, model, model_name, arch_type,
    device, dataset, base_loss_fn,  
    optimizer, patience=5
):
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)
    dataset_name = dataset['info']['id'].split('--')[0]
    task_type = dataset['info']['task_type']

    train_times = []
    val_times = [] # времена инференса можно замерять и на val!

    val_best_epoch = {'epoch' : 0, 'acc' : 0, 'loss' : 10**20}
    test_best_epoch = {'epoch' : 0, 'acc' : 0, 'loss' : 10**20}

    remaining_patience = patience
    total_epochs = 0
    for epoch in tqdm(range(epochs), desc = f'{model_name}_{arch_type} on {dataset_name}'):
        total_epochs +=1 

        train_loss, train_acc, train_time = train_epoch(model, device, dataset, base_loss_fn, optimizer, scheduler, model_name, arch_type)
        # тестируем и валидируем
        val_loss, val_acc, val_time = validate(model, device, dataset, base_loss_fn, 'val', model_name, arch_type)
        test_loss, test_acc, test_time = validate(model, device, dataset, base_loss_fn, 'test', model_name, arch_type)

        train_times.append(train_time)
        val_times.append(val_time)


        # обновляем лучшие эпохи для val/test
        val_epoch = {'epoch' : epoch + 1, 'loss' : val_loss, 'acc' : val_acc}
        test_epoch = {'epoch' : epoch + 1, 'loss' : test_loss, 'acc' : test_acc}
        if compare_epochs(task_type, val_epoch, val_best_epoch):
            val_best_epoch = val_epoch
            remaining_patience = patience
            test_real_epoch = test_epoch
        else:
            remaining_patience -= 1
        if remaining_patience < 0:
            break
        if compare_epochs(task_type, test_epoch, test_best_epoch):
            test_best_epoch = test_epoch


    final_logs = {
        'train_epoch_time' : sum(train_times) / total_epochs,
        'val_epoch_time' : sum(val_times) / total_epochs,
        'full_train_time' : sum(train_times),
        'num_epochs' : total_epochs,
        'num_params' : count_parameters(model),
        'in_features' : dataset['info']['in_features'],
        'out_features' : (1 if task_type != 'multiclass' else dataset['info']['n_classes']),
        'val_best_epoch' : val_best_epoch['epoch'],
        'val_best_loss' : val_best_epoch['loss'],
        'test_best_epoch' : test_best_epoch['epoch'],
        'test_best_loss' : test_best_epoch['loss']
        # ширины и так будут залоггированы
    }
    if task_type != 'regression':
        final_logs.update({
            'val_best_acc' : val_best_epoch['acc'],
            'test_best_acc' : test_best_epoch['acc']
        })

    return final_logs
