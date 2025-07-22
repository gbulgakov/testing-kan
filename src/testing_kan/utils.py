import zipfile
import os
import gc
import yaml
import torch
from .optimizers.ademamix import AdEMAMix
from .optimizers.muon import Muon
from .optimizers.mars import MARS
from .optimizers.sign import Signum
# удобно для масшатбирования
OPTIMIZERS = { 
              'adamw' : torch.optim.AdamW,
              'ademamix' : AdEMAMix,
              'mars' : MARS,
              'muon' : Muon,
              'sign' : Signum,
              'momentum_sign' : Signum
             }

def create_zip_archive(source_dir, archive_path):
    """
    Собирает все файлы из директории source_dir в один ZIP-архив.

    Args:
        source_dir (str): Путь к папке с файлами для архивации.
        archive_path (str): Полный путь для сохранения ZIP-архива.
    """
    # Используем 'w' для создания нового архива и ZIP_DEFLATED для сжатия
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir(source_dir):
            # Полный путь к файлу, который мы добавляем
            file_path = os.path.join(source_dir, filename)
            # Убедимся, что это файл, а не папка, и что это не сам будущий архив
            if os.path.isfile(file_path) and not filename.endswith('.zip'):
                # Добавляем файл в архив. arcname=filename гарантирует,
                # что в архиве не будет лишних путей (только имя файла).
                zipf.write(file_path, arcname=filename)

def load_secrets():
    """Загружает секреты из YAML-файла."""
    current_file_path = os.path.abspath(__file__)

    # Шаг 2: Находим папку, где лежит utils.py
    # os.path.dirname() "отрезает" имя файла и оставляет только путь к папке.
    current_dir = os.path.dirname(current_file_path)

    # Шаг 3: Поднимаемся на два уровня вверх, чтобы добраться до корня проекта
    # utils.py -> testing_kan -> src -> testing-kan (корень)
    # Нам нужно подняться от папки `testing_kan` и от папки `src`
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # Шаг 4: Строим полный путь к файлу secrets.yaml
    secrets_path = os.path.join(project_root, 'secrets.yaml')
    with open(secrets_path, 'r') as f:
        data = yaml.safe_load(f)

        # Получаем список словарей
    telegram_list = data.get('telegram', [])
    
    # Преобразуем список в один словарь
    telegram_secrets = {}
    for item in telegram_list:
        telegram_secrets.update(item)
    
    # Возвращаем данные в удобном формате
    return {'telegram': telegram_secrets}


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

# True, если эпоха 1 лучше эпохи 2
def compare_epochs(task_type, epoch_dict1, epoch_dict2):
    if task_type == 'regression':
        return epoch_dict1['loss'] < epoch_dict2['loss']
    else:
        return epoch_dict1['acc'] > epoch_dict2['acc']
    
def get_optimizer(optim_name, model_params, config):
    optim_class = OPTIMIZERS[optim_name]
    optim_kwargs = {'lr' : config['lr']}

    if optim_name != 'muon':             # для muon все параметры  -- muon_params
        optim_kwargs['weight_decay'] = config['weight_decay']
    if optim_name == 'momentum_sign':    # базово делаем без моментума
        optim_kwargs['momentum'] = 0.9

    if optim_name == 'muon':
        model_params = list(model_params)
    return optim_class(model_params, **optim_kwargs)
 
def clean_up_model(model, optimizer):
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

