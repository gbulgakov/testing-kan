import sys
import os
from pathlib import Path

HOME = str(Path.home())
sys.path.append(os.path.join(HOME, 'KAN', 'testing-kan'))

from IPython.display import clear_output

from src.testing_kan import data_processing
from src.testing_kan.tg_bot import send_telegram_file, send_telegram_message
from src.testing_kan.pipelines.tuning import tune
from src.testing_kan.pipelines.testing import test_best_model
from src.testing_kan.logger import Logger
from src.testing_kan.utils import create_zip_archive

from functools import wraps
import traceback
from datetime import datetime

def telegram_error_notification(func):
    """Декоратор для отправки ошибок в Telegram"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            
            # Формируем подробное сообщение
            error_msg = (
                f"*❌ Ошибка*\n\n"
                f"*Model:* `{kwargs.get('model_name', 'N/A')}`\n"
                f"*Dataset:* `{kwargs.get('dataset_name', 'N/A')}`\n"
                f"*Arch type:* `{kwargs.get('arch_type', 'N/A')}`\n"
                f"*Embedding:* `{kwargs.get('emb_name', 'N/A')}`\n"
                f"*Optimizer:* `{kwargs.get('optim_name', 'N/A')}`\n"
                f"*Функция:* `{func.__name__}`\n\n"
                f"*Ошибка:*\n```\n{str(e)}\n```\n\n"
                f"*Traceback:* (нажмите ▶️)\n"
                f"```\n{traceback.format_exc()}\n```"
            )
            send_telegram_message(error_msg)
            raise  # Пробрасываем исключение дальше
    return wrapper

@telegram_error_notification 
def run_single_model(
        *,
        model_name, 
        arch_type, 
        emb_name, 
        optim_name, 
        dataset, 
        num_epochs, 
        num_trials, 
        patience
):
    best_params = tune(
            model_name=model_name,
            arch_type=arch_type,
            emb_name=emb_name,
            optim_name=optim_name,
            dataset=dataset,
            num_epochs=num_epochs,
            num_trials=num_trials,
            patience=patience
        )
    clear_output(wait=True)
    stats = test_best_model(
        best_params=best_params,
        model_name=model_name,
        arch_type=arch_type,
        emb_name=emb_name,
        optim_name=optim_name,
        dataset=dataset,
        num_epochs=num_epochs,
        patience=patience
    )
    return stats

def run_single_dataset(
        *, 
        dataset_name,
        optim_names, 
        emb_names, 
        model_names, 
        arch_types,
        num_epochs, 
        num_trials, 
        patience, 
        logger
):
    # dataset_type = dataset_info['type']
    zip_path = os.path.join(HOME, 'KAN', 'data', f'{dataset_name}.zip')
    dataset = data_processing.load_dataset(dataset_name, zip_path)

    for model_name in model_names:
        for arch_type in arch_types:
            for optim_name in optim_names:
                for emb_name in emb_names:
                    # Аргументы в run_single_model можно немного почистить
                    stats = run_single_model(
                        model_name=model_name,
                        arch_type=arch_type,
                        emb_name=emb_name,
                        optim_name=optim_name,
                        dataset=dataset,
                        num_epochs=num_epochs,
                        num_trials=num_trials,
                        patience=patience
                    )
                    clear_output(wait=True)
                    
                    # 3. Вызываем метод логгера с отдельными параметрами
                    logger.log_run(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        arch_type=arch_type,
                        emb_name=emb_name,
                        optim_name=optim_name,
                        stats=stats
                    )


def run_experiment(
    *,
    project_name,
    dataset_names,
    model_names,
    emb_names,
    optim_names,
    arch_types,
    num_epochs,
    num_trials,
    patience,
    exp_name
):
    send_telegram_message(
        f"🚀 *Запуск эксперимента*\n"
        f"▫️ *Название:* `{exp_name}`\n"
        f"▫️ *Время:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
    )
    # Logger
    results_dir = os.path.join(HOME, 'KAN', 'testing-kan', 'results')
    logger = Logger(results_dir, exp_name)

    # Запуски
    for dataset_name in dataset_names:
        run_single_dataset(
            project_name=project_name,
            dataset_name=dataset_name,
            optim_names=optim_names,
            emb_names=emb_names,
            model_names=model_names,
            arch_types=arch_types,
            num_epochs=num_epochs,
            num_trials=num_trials,
            patience=patience,
            logger=logger  # <--- Передаем логгер
        )

    # В конце вызываем один метод, который сохранит все файлы (JSON и CSV)
    logger.save()

    # Создаем общий ZIP-архив
    archive_name = f'{exp_name}_logs.zip'
    archive_path = os.path.join(results_dir, archive_name)
    create_zip_archive(source_dir=results_dir, archive_path=archive_path)

    # Отправляем в Telegram именно ZIP-архив
    send_telegram_file(archive_path)
    
    # Можно обновить финальное сообщение
    send_telegram_message(f'✅ Эксперимент {exp_name} завершен. Архив с логами отправлен.')

