import os
import json
import pandas as pd
from collections import defaultdict

class Logger:
    """
    Класс для логирования результатов экспериментов в форматах JSON и CSV для бустинговых моделей.
    """
    def __init__(self, results_dir, exp_name):
        self.results_dir = results_dir
        self.exp_name = exp_name
        self.raw_results = []
        os.makedirs(self.results_dir, exist_ok=True)

        # Карта отображения названий датасетов и информации о метрике (↑ — выше лучше, ↓ — ниже).
        self.dataset_column_map = {
            'adult': ('adult ↑', True),
            'gesture': ('gesture ↑', True),
            'california': ('california ↓', False),
            'churn': ('churn ↑', True),
            'house': ('house ↓', False),
            'fb-comments': ('fb-comments ↓', False),
            'otto': ('otto ↑', True),
            'ecom-offers': ('ecom-offers ↑', True),
            'microsoft': ('microsoft ↓', False),
            'santander': ('santander ↑', True),
            'black-friday': ('black-friday ↓', False),
            'covtype': ('covtype ↑', True),
            'higgs-small': ('higgs-small ↑', True),
            'diamond': ('diamond ↓', False),
            'regression-num-large-0-year': ('regression-num-large-0-year ↓', False),
            'regression-cat-large-0-particulate-matter-ukair-2017': ('regression-cat-large-0-particulate-matter-ukair-2017 ↓', False),
            'regression-cat-large-0-nyc-taxi-green-dec-2016': ('regression-cat-large-0-nyc-taxi-green-dec-2016 ↓', False),
            'classif-cat-large-0-road-safety': ('classif-cat-large-0-road-safety ↑', True),
            'regression-num-medium-0-medical_charges': ('regression-num-medium-0-medical_charges ↓', False),
            'classif-num-large-0-MiniBooNE': ('classif-num-large-0-MiniBooNE ↑', True)
        }

    def log_run(self, dataset_name, model_name, stats):
        """Добавляет результаты одного запуска в список."""
        self.raw_results.append({
            "dataset": dataset_name,
            "model": model_name,
            **stats
        })

    def save(self):
        """Сохраняет все результаты в JSON и CSV."""
        if not self.raw_results:
            print("Нет данных для сохранения.")
            return

        self._save_as_json()
        self._save_as_csv()
        print(f"Результаты эксперимента '{self.exp_name}' успешно сохранены в {self.results_dir}")

    def _make_json_safe(self, obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(x) for x in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        return obj
    
    def _save_as_json(self):
        """Сохраняет полные результаты в JSON, добавляя к существующим данным вместо перезаписи."""
        import json

        json_path = os.path.join(self.results_dir, f'{self.exp_name}_results.json')

        # Загружаем существующие данные, если файл есть
        if os.path.isfile(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            existing_data = {}

        # Преобразуем existing_data в defaultdict(list) для удобства добавления
        combined_data = defaultdict(list)
        # Заполняем из существующих данных
        for dataset_name, runs in existing_data.items():
            combined_data[dataset_name].extend(runs)

        # Добавляем текущие результаты self.raw_results
        for run in self.raw_results:
            combined_data[run['dataset']].append(run)

        # Сохраняем обратно, приводя defaultdict к dict
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self._make_json_safe(dict(combined_data)), f, ensure_ascii=False, indent=4)

    def _save_as_csv(self):
        """Создает CSV файлы по метрике, времени обучения и инференса."""
        json_path = os.path.join(self.results_dir, f'{self.exp_name}_results.json')

        # Если JSON файл есть — загрузить данные из него
        if os.path.isfile(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    loaded_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: JSON файл {json_path} поврежден или пуст, используем текущие raw_results")
                    loaded_data = None
            # Преобразуем загруженный словарь в список словарей (flat list), как raw_results
            if loaded_data is not None:
                all_runs = []
                for runs in loaded_data.values():
                    all_runs.extend(runs)
                df = pd.DataFrame(all_runs)
            else:
                df = pd.DataFrame(self.raw_results)
        else:
            df = pd.DataFrame(self.raw_results)

        if df.empty:
            print("Нет данных для создания CSV.")
            return

        # Обобщённая колонка модели (у нас только model)
        df['Model'] = df['model']

        # Строка "метрика ± std"
        df['performance_str'] = df.apply(
            lambda row: f"{row['metric']:.3f} + {row['metric_std']:.3f}", axis=1)

        df['train_time_str'] = df.apply(
            lambda row: f"{row['full_train_time']:.3f} + {row['full_train_time_std']:.3f}", axis=1)

        df['inference_time_str'] = df.apply(
            lambda row: f"{row['full_val_time']:.3f} + {row['full_val_time_std']:.3f}", axis=1)

        # Красивые имена датасетов
        df['dataset_pretty'] = df['dataset'].apply(
            lambda x: self.dataset_column_map.get(x, (x, True))[0]
        )

        sorted_columns = [info[0] for info in self.dataset_column_map.values()]

        # Сохраняем три таблицы
        self._create_pivot_and_save(df, 'performance_str', 'results.csv', sorted_columns)
        self._create_pivot_and_save(df, 'train_time_str', 'train_time.csv', sorted_columns)
        self._create_pivot_and_save(df, 'inference_time_str', 'inference_time.csv', sorted_columns)

    def _create_pivot_and_save(self, df, value_col, filename, columns_order):
        pivot_df = df.pivot_table(
            index='Model',
            columns='dataset_pretty',
            values=value_col,
            aggfunc='first'
        )
        ordered_cols_exist = [col for col in columns_order if col in pivot_df.columns]
        pivot_df = pivot_df[ordered_cols_exist]

        csv_path = os.path.join(self.results_dir, f'{self.exp_name}_{filename}')
        pivot_df.to_csv(csv_path)
