from exp_runner import run_experiment
import argparse
import yaml


# парсинг аргументов командной строки для запуска
def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with config and CLI override")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config")
    parser.add_argument('--datasets', nargs='+', help="List of datasets names")
    parser.add_argument('--models', nargs='+', help="List of model names")
    parser.add_argument('--embs', nargs='+', help="List of embeddings")
    parser.add_argument('--optimizers', nargs='+', help="List of optimizers")
    parser.add_argument('--arch_types', nargs='+', help="List of architectures")
    parser.add_argument('--exp-name', type=str, required=True, help="Name of this experiment")

    return parser.parse_args()


def merge_config_with_cli(config, args):
    """
    Объединяет конфиг из YAML с аргументами CLI.
    Значения из CLI имеют приоритет над значениями из конфига.
    """
    merged = config.copy()

    merged["exp_name"] = args.exp_name
    
    if args.datasets is not None:
        merged["dataset_names"] = args.datasets
    if args.models is not None:
        merged["model_names"] = args.models
    if args.embs is not None:
        merged["emb_names"] = args.embs
    if args.optim_names is not None:
        merged["optim_names"] = args.optim_names
    if args.arch_types is not None:
        merged["arch_types"] = args.arch_types

    return merged


def main():
    args = parse_args()
    
    # Загружаем конфиг из YAML
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Объединяем конфиг с аргументами CLI
    final_config = merge_config_with_cli(config, args)

    # Запускаем эксперимент
    run_experiment(**final_config)

if __name__ == "__main__":
    main()