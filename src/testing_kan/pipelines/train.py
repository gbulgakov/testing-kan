import time
from typing import Dict, Any, Callable, Tuple, List
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from src.testing_kan.utils import count_parameters, compare_epochs


def get_loss_fn(
    arch_type: str,
    base_loss_fn: Callable[[Tensor, Tensor], Tensor],
    task_type: str,
    share_training_batches: bool
) -> Callable[[Tensor, Tensor], Tensor]:
    """
    Create a loss function wrapper based on architecture type and task configuration.
    """
    if arch_type != 'plain':
        loss_fn = lambda y_pred, y_true: base_loss_fn(
            y_pred.flatten(0, 1),
            y_true.repeat_interleave(y_pred.shape[-1 if task_type != 'multiclass' else -2]) if share_training_batches else y_true,
        )
    else:
        loss_fn = lambda y_pred, y_true: base_loss_fn(y_pred.squeeze(1), y_true)
    return loss_fn


# Automatic mixed precision (AMP)
# torch.bfloat16 is preferred over torch.float16 for better numerical stability
amp_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
    if torch.cuda.is_available()
    else None
)

# AMP is disabled by default - set to True for faster training on compatible hardware
amp_enabled = False and amp_dtype is not None
device = torch.device('cuda')


@torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
def apply_model(batch_data: Dict[str, Tensor], model: nn.Module) -> Tensor:
    """Apply the model to the batch data."""
    return (
        model(
            batch_data['X_num'],
            batch_data.get('X_cat')
        )
        .squeeze(-1)  # Remove the last dimension for regression tasks.
        .float()
    )


def train_epoch(
        model: nn.Module, 
        device: torch.device, 
        dataset: dict, 
        base_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
        optimizer: Optimizer, 
        scheduler: _LRScheduler, 
        arch_type: str
    ) -> Tuple[float, float, float]:
    '''
    Train the model for one epoch
    '''

    task_type = dataset['info']['task_type']
    loader = dataset['train']

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
        
        # Forward pass: output shape is (B, k, n_out) or (B, k) for regression/binclass
        output = apply_model(batch_data, model)
        loss_value = loss_fn(output, batch_data['y'])
        loss_value.backward()
        optimizer.step()
        train_loss += loss_value.item()
        
        # Calculate accuracy based on whether training batches are shared
        if model.share_training_batches:
            output = output.mean(dim=1)  # Average over ensemble members: (B, n_out) or (B)
            y_true = batch_data['y']     # Ground truth: (B)
        else:
            output = output              # Keep ensemble dimension: (B, k, n_out) or (B, k)
            y_true = batch_data['y']     # Ground truth: (B, k)
        
        
        # Generate predictions (simplified approach for all task types)
        if task_type == 'binclass' or task_type == 'regression':
            pred = (output >= 0).float()  # Binary threshold at 0.0
        else:
            pred = output.argmax(1)       # Multiclass: argmax over classes

        correct += (pred == y_true).float().sum().item()
        total += y_true.numel()

    scheduler.step()

    end_time = time.time()
    epoch_time = end_time - start_time

    num_batches = len(loader)
    train_accuracy = correct / total

    return train_loss / num_batches, train_accuracy, epoch_time 
    
def validate(
        model: nn.Module, 
        device: torch.device, 
        dataset:  Dict[str, Any], 
        base_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        part: str, 
        model_name: str, 
    ) -> Tuple[float, float, float]:
    '''
    Validate the model on a specified dataset partition.
    '''
    loader = dataset[part]
    model.eval()
    model.to(device)
    
    val_loss = 0.0
    correct = 0
    total = 0
    task_type = dataset['info']['task_type']

    task_type = dataset['info']['task_type']
    
    loss_fn = get_loss_fn(model_name, base_loss_fn, task_type, model.share_training_batches)
    
    with torch.no_grad():
        start_time = time.time()

        for (X_num, X_cat, y) in loader:
            X_num = X_num.to(device)
            X_cat = X_cat.to(device)
            y = y.to(device)
            
            batch_data = {
                'X_num': X_num,
                'X_cat': X_cat if X_cat.size(1) > 0 else None,
                'y': y
            }
            
            # Forward pass: output shape is (B, k, n_out) or (B, k)
            output = apply_model(batch_data, model)
            val_loss += loss_fn(output, batch_data['y']).item()
            
            # Average predictions over ensemble members for final prediction
            output = output.mean(dim=1)  # (B, n_out) or (B)

            # Average predictions over ensemble members for final prediction
            output = output.mean(dim=1)  # (B, n_out) or (B)
            
            # Generate predictions (simplified approach for all task types)
            if task_type == 'binclass' or task_type == 'regression':
                pred = (output >= 0).float()
            else:
                pred = output.argmax(1)  # Multiclass
                
            correct += (pred == batch_data['y']).float().sum().item()
            total += batch_data['y'].numel()

        end_time = time.time()
        val_time = end_time - start_time
        
    num_batches = len(loader)
    val_accuracy = correct / total

    return val_loss / num_batches, val_accuracy, val_time

def train(
    epochs: int,
    model: nn.Module, 
    model_name: str, 
    arch_type: str,
    device: torch.device, 
    dataset: Dict[str, Any], 
    base_loss_fn: Callable[[Tensor, Tensor], Tensor],  
    optimizer: Optimizer, 
    patience: int=5
) -> Dict[str, Any]:
    '''
    Train the model for multiple epochs with early stopping based on validation performance.
    '''
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)

    dataset_name = dataset['info']['id'].split('--')[0]
    task_type = dataset['info']['task_type']

    train_times = []
    val_times = [] # inference times are measured on val
    val_best_epoch = {'epoch' : 0, 'acc' : 0, 'loss' : 10**20}
    test_best_epoch = {'epoch' : 0, 'acc' : 0, 'loss' : 10**20}
    remaining_patience = patience
    total_epochs = 0

    for epoch in tqdm(range(epochs), desc = f'{model_name}_{arch_type} on {dataset_name}'):
        total_epochs +=1 

        # Training step
        train_loss, train_acc, train_time = train_epoch(
            model, device, dataset, base_loss_fn, optimizer, scheduler, arch_type
        )

        # Validation and test evaluation
        val_loss, val_acc, val_time = validate(
            model, device, dataset, base_loss_fn, 'val', model_name
        )
        test_loss, test_acc, test_time = validate(
            model, device, dataset, base_loss_fn, 'test', model_name
        )

        train_times.append(train_time)
        val_times.append(val_time)

        # Track best epochs for validation and test sets
        val_epoch = {'epoch': epoch + 1, 'loss': val_loss, 'acc': val_acc}
        test_epoch = {'epoch': epoch + 1, 'loss': test_loss, 'acc': test_acc}
        
        # Early stopping logic based on validation performance
        if compare_epochs(task_type, val_epoch, val_best_epoch):
            val_best_epoch = val_epoch
            remaining_patience = patience  # Reset patience counter
        else:
            remaining_patience -= 1
            
        if remaining_patience < 0:
            break
            
        # Track best test performance (for reporting, not early stopping)
        if compare_epochs(task_type, test_epoch, test_best_epoch):
            test_best_epoch = test_epoch

    # Compile final training logs
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
    }

    # Add accuracy metrics for classification tasks
    if task_type != 'regression':
        final_logs.update({
            'val_best_acc' : val_best_epoch['acc'],
            'test_best_acc' : test_best_epoch['acc']
        })

    return final_logs