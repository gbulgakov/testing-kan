import sys
import os
from pathlib import Path
from turtle import mode

HOME = str(Path.home())
sys.path.append(os.path.join(HOME, 'KAN', 'testing-kan'))

import numpy as np
import pickle
import wandb
from IPython.display import clear_output
# wandb.login(key='936d887040f82c8da3d87f5207c4178259c7b922')
from gbdt_utils import load_dataset
from src.testing_kan.utils import create_zip_archive
from src.testing_kan.tg_bot import send_telegram_file, send_telegram_message
from src.testing_kan.logger import Logger
from tuning import tune
from testing import test_best_model
# from wandb_tuning import wandb_tuning
# from wandb_testing import test_best_model

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
                f"*Функция:* `{func.__name__}`\n\n"
                f"*Ошибка:*\n```\n{str(e)}\n```\n\n"
                f"*Traceback:* (нажмите ▶️)\n"
                f"```\n{traceback.format_exc()}\n```"
            )
            send_telegram_message(error_msg)
            raise  # Пробрасываем исключение дальше
    return wrapper


def run_single_model(project_name, dataset_name, model_name, dataset, num_trials):
    best_params = tune(
        model_name=model_name,
        dataset=dataset,
        num_trials=num_trials
    )
    clear_output(wait=True)
    stats = test_best_model(
        best_params=best_params,
        project_name=project_name,
        dataset_name=dataset_name,
        model_name=model_name,
        dataset=dataset)
    return stats

def run_single_dataset(project_name, dataset_name, model_names, num_trials, logger):
    # dataset_type = dataset_info['type']
    zip_path = os.path.join(HOME, 'KAN', 'data', f'{dataset_name}.zip')
    dataset = load_dataset(dataset_name, zip_path)

    for model_name in model_names:
        stats = run_single_model(project_name, dataset_name, model_name, dataset, num_trials=num_trials)
        clear_output(wait=True)

        logger.log_run(
            dataset_name=dataset_name,
            model_name=model_name,
            stats=stats
        )


def run_experiment(
    *,
    project_name,
    dataset_names,
    model_names,
    num_trials,
    exp_name
):
    send_telegram_message(
        f"🚀 *Запуск эксперимента*\n"
        f"▫️ *Название:* `{exp_name}`\n"
        f"▫️ *Время:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
    )

    # Logger
    results_dir = os.path.join(HOME, 'KAN', 'testing-kan', 'results', exp_name)
    logger = Logger(results_dir, exp_name)

    # Создаем директорию для результатов, если ее нет
    for dataset_name in dataset_names:
        run_single_dataset(
            project_name=project_name,
            dataset_name=dataset_name,
            model_names=model_names,
            num_trials=num_trials,
            logger=logger
            )
        logger.save()
        # send_telegram_message(f'✅ Finished on {dataset_name}')

    # Создаем общий ZIP-архив
    archive_name = f'{exp_name}_logs.zip'
    archive_path = os.path.join(results_dir, archive_name)
    create_zip_archive(source_dir=results_dir, archive_path=archive_path)

    # Отправляем в Telegram именно ZIP-архив
    send_telegram_file(archive_path)
    
    # Можно обновить финальное сообщение
    send_telegram_message(f'✅ Эксперимент {exp_name} завершен. Архив с логами отправлен.')

