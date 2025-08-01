import os
import json
import pandas as pd
from collections import defaultdict

class Logger:
    '''
    Class that loggs experiments in JSON and CSV formats.
    It collects all the data and save it after all runs
    '''
    def __init__(self, results_dir, exp_name):
        self.results_dir = results_dir
        self.exp_name = exp_name
        self.raw_results = []
        os.makedirs(self.results_dir, exist_ok=True)

        # key: dataset name
        # valye: (final column name, metric: up is better?)
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
        '''Collects info from one full run (tuning + evaluation)'''
        self.raw_results.append({
            "dataset": dataset_name,
            "model": model_name,
            "emb": emb_name,
            "arch_type": arch_type,
            "optimizer": optim_name,
            **stats
        })

    def save(self):
        '''Save everything in JSON and CSV'''
        """Сохраняет все собранные результаты в файлы JSON и CSV."""
        if not self.raw_results:
            print("No data for saving.")
            return

        self._save_as_json()
        self._save_as_csv()
        print(f"Experiment results'{self.exp_name}' successfully saved in {self.results_dir}")

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
        '''
        Save all the results (grouped by dataset name) in JSON
        '''
 
        json_path = os.path.join(self.results_dir, f'{self.exp_name}_results.json')
        
        # group by key 'dataset'
        results_by_dataset = defaultdict(list)
        for run_data in self.raw_results:
            dataset_name = run_data.get('dataset', 'unknown_dataset')
            results_by_dataset[dataset_name].append(run_data)

        # convert to classic dict and make datatypes safe for JSON
        final_data = self._make_json_safe(dict(results_by_dataset))

        # finally saving
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        
        print(f"-> JSON-отчет сохранен в: {json_path}")


    def _save_as_csv(self, model_keys = ['model', 'emb', 'optimizer', 'arch_type']):
        '''
        Generate a unified DataFrame of all results and then generate 
        three summary CSV-tables via pivot_table
        '''
        
        json_path = os.path.join(self.results_dir, f'{self.exp_name}_results.json')

        # loading from JSON if possible
        if os.path.isfile(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    loaded_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: JSON file {json_path} hurt of empty, so using raw_results")
                    loaded_data = None
            # converting loaded dict into в list of dicts(flat list), like raw_results
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
            print("No data for CSV")
            return

        # 2. Preparing data for summary tables
        
        # Creating combined index column
        df['Model'] = df['model'] + '_' + df['arch_type']
        df['Model'] = df[model_keys[0]].str.cat(df[model_keys[1:]], sep='_')

        # Creating formatted columnb "avg +- std"
        df['performance_str'] = df.apply(
            lambda row: f"{row['metric']:.3f} + {row['metric_std']:.3f}", axis=1)
        df['train_time_str'] = df.apply(
            lambda row: f"{row['full_train_time']:.3f} + {row['full_train_time_std']:.3f}", axis=1)
        df['inference_time_str'] = df.apply(
            lambda row: f"{row['val_epoch_time']:.3f} + {row['val_epoch_time_std']:.3f}", axis=1)

        # pretty dataset names
        df['dataset_pretty'] = df['dataset'].apply(
            lambda x: self.dataset_column_map.get(x, (x, True))[0])
        
        # needed column order
        sorted_columns = [info[0] for info in self.dataset_column_map.values()]

        # 3. creating summary table.
        self._create_pivot_and_save(df, 'performance_str', 'results.csv', sorted_columns)
        self._create_pivot_and_save(df, 'train_time_str', 'train_time.csv', sorted_columns)
        self._create_pivot_and_save(df, 'inference_time_str', 'inference_time.csv', sorted_columns)

    def _create_pivot_and_save(self, df, value_col, filename, columns_order):
        """
        Creating (pivot table) and saving as CSV.
        
        Args:
            df (pd.DataFrame): originil DataFrame with data
            value_col (str): column name, which values go to table
            filename (str) :output csv name
            columns_order (list): needed column order.
        """
        # creating pivot table
        # - index: 'Model'.
        # - columns: from 'dataset_pretty'.
        # - Cells: from  value_col.
        pivot_df = df.pivot_table(
            index='Model', 
            columns='dataset_pretty', 
            values=value_col, 
            aggfunc='first'  # (Model, dataset) pairs are unique
        )
        
        # reordering columns
        ordered_cols_exist = [col for col in columns_order if col in pivot_df.columns]
        pivot_df = pivot_df[ordered_cols_exist]

        csv_path = os.path.join(self.results_dir, f'{self.exp_name}_{filename}')
        pivot_df.to_csv(csv_path)
