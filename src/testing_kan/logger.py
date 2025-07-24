import os
import json
import pandas as pd
from collections import defaultdict

class Logger:
    """
    Класс для логирования результатов экспериментов в форматах JSON и CSV.
    Собирает данные по ходу выполнения и сохраняет их в конце.
    """
    def __init__(self, results_dir, exp_name):
        self.results_dir = results_dir
        self.exp_name = exp_name
        self.raw_results = []
        os.makedirs(self.results_dir, exist_ok=True)

        # костыль для удобства        
        # Ключ: внутреннее имя датасета.
        # Значение: кортеж (публичное_имя_колонки, метрика_вверх_лучше?)
        self.dataset_column_map = {
            'adult': ('adult ↑', True),
            'gesture': ('gesture ↑', True),
            'california': ('california ↓', False),
            'churn': ('churn ↑', True),
            'house': ('house ↓', False),
            'fb-comments': ('fb-comments ↓', False),
            'otto': ('otto ↑', True), # <-- Изменено на ↑
            'ecom-offers': ('ecom-offers ↑', True),
            'microsoft': ('microsoft ↓', False), # <-- Изменено на ↓
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
    

    def log_run(self, dataset_name, model_name, emb_name, arch_type, optim_name, stats):
        """Собирает результаты одного полного запуска (тюнинг + тест)."""
        self.raw_results.append({
            "dataset": dataset_name,
            "model": model_name,
            "emb": emb_name,
            "arch_type": arch_type,
            "optimizer": optim_name,
            **stats
        })

    def save(self):
        """Сохраняет все собранные результаты в файлы JSON и CSV."""
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
        """Сохраняет полные результаты в JSON, заменяя старые записи новыми по ключам (dataset, model)."""
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

        # Преобразуем existing_data в словарь для быстрого поиска по (dataset, model)
        # Структура: {dataset: {model: run_dict}}
        indexed_data = {}

        for dataset_name, runs in existing_data.items():
            indexed_data.setdefault(dataset_name, {})
            for run in runs:
                model_name = run.get('model')
                if model_name is not None:
                    indexed_data[dataset_name][model_name] = run

        # Теперь проходим по новым результатам и заменяем или добавляем
        for run in self.raw_results:
            dataset_name = run['dataset']
            model_name = run.get('model')
            if dataset_name not in indexed_data:
                indexed_data[dataset_name] = {}
            # Заменяем старую запись или добавляем новую
            indexed_data[dataset_name][model_name] = run

        # Преобразуем обратно в структуру {dataset: list_of_runs}
        combined_data = {ds: list(models.values()) for ds, models in indexed_data.items()}

        # Сохраняем обратно в файл, приводя данные к json-friendly типам
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self._make_json_safe(combined_data), f, ensure_ascii=False, indent=4)

    
    # def _save_as_json(self):
    #     """Сохраняет полные результаты в JSON, добавляя к существующим данным вместо перезаписи."""
    #     import json

    #     json_path = os.path.join(self.results_dir, f'{self.exp_name}_results.json')

    #     # Загружаем существующие данные, если файл есть
    #     if os.path.isfile(json_path):
    #         with open(json_path, 'r', encoding='utf-8') as f:
    #             try:
    #                 existing_data = json.load(f)
    #             except json.JSONDecodeError:
    #                 existing_data = {}
    #     else:
    #         existing_data = {}

    #     # Преобразуем existing_data в defaultdict(list) для удобства добавления
    #     combined_data = defaultdict(list)
    #     # Заполняем из существующих данных
    #     for dataset_name, runs in existing_data.items():
    #         combined_data[dataset_name].extend(runs)

    #     # Добавляем текущие результаты self.raw_results
    #     for run in self.raw_results:
    #         combined_data[run['dataset']].append(run)

    #     # Сохраняем обратно, приводя defaultdict к dict
    #     with open(json_path, 'w', encoding='utf-8') as f:
    #         json.dump(self._make_json_safe(dict(combined_data)), f, ensure_ascii=False, indent=4)

    def _save_as_csv(self, model_keys = ['model', 'emb']):
        """
        Создает единый DataFrame из всех результатов и на его основе
        генерирует три сводные CSV-таблицы с помощью pivot_table.
        """
        
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

        # 2. Подготавливаем данные для сводных таблиц.
        
        # Создаем комбинированную колонку 'Model' для использования в качестве индекса.
        # Это место можно легко настроить, если вы захотите добавить, например, optim_name в индекс.
        df['Model'] = df['model'] + '_' + df['arch_type']
        df['Model'] = df[model_keys[0]].str.cat(df[model_keys[1:]], sep='_')

        # Создаем колонки с отформатированными строками "среднее ± стд. отклонение".
        df['performance_str'] = df.apply(
            lambda row: f"{row['metric']:.3f} + {row['metric_std']:.3f}", axis=1)
        df['train_time_str'] = df.apply(
            lambda row: f"{row['full_train_time']:.3f} + {row['full_train_time_std']:.3f}", axis=1)
        df['inference_time_str'] = df.apply(
            lambda row: f"{row['val_epoch_time']:.3f} + {row['val_epoch_time_std']:.3f}", axis=1)

        # Создаем "красивые" названия колонок для датасетов.
        df['dataset_pretty'] = df['dataset'].apply(
            lambda x: self.dataset_column_map.get(x, (x, True))[0])
        
        # Определяем желаемый порядок колонок в итоговых таблицах.
        sorted_columns = [info[0] for info in self.dataset_column_map.values()]

        # 3. Создаем и сохраняем каждую сводную таблицу.
        self._create_pivot_and_save(df, 'performance_str', 'results.csv', sorted_columns)
        self._create_pivot_and_save(df, 'train_time_str', 'train_time.csv', sorted_columns)
        self._create_pivot_and_save(df, 'inference_time_str', 'inference_time.csv', sorted_columns)

    def _create_pivot_and_save(self, df, value_col, filename, columns_order):
        """
        Создает сводную таблицу (pivot table) и сохраняет ее в CSV.
        
        Args:
            df (pd.DataFrame): Исходный DataFrame со всеми данными.
            value_col (str): Название колонки, значения из которой пойдут в ячейки таблицы.
            filename (str): Имя выходного CSV файла.
            columns_order (list): Желаемый порядок колонок.
        """
        # Создаем сводную таблицу:
        # - Строки (индекс) будут из колонки 'Model'.
        # - Колонки будут из 'dataset_pretty'.
        # - Ячейки будут заполнены значениями из value_col.
        pivot_df = df.pivot_table(
            index='Model', 
            columns='dataset_pretty', 
            values=value_col, 
            aggfunc='first'  # Так как каждая пара (Model, dataset) уникальна, aggfunc просто берет одно значение.
        )
        
        # Приводим колонки к правильному порядку, если они существуют в таблице.
        # Приводим колонки к правильному порядку
        ordered_cols_exist = [col for col in columns_order if col in pivot_df.columns]
        pivot_df = pivot_df[ordered_cols_exist]

        csv_path = os.path.join(self.results_dir, f'{self.exp_name}_{filename}')
        pivot_df.to_csv(csv_path)