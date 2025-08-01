import argparse
import yaml
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

from run_experiment import run_experiment

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with config and CLI override")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config")
    parser.add_argument('--datasets', nargs='+', help="List of datasets names")
    parser.add_argument('--models', nargs='+', help="List of model names")
    parser.add_argument('--embs', nargs='+', help="List of embeddings")
    parser.add_argument('--optim-names', nargs='+', help="List of optimizers")
    parser.add_argument('--arch-types', nargs='+', help="List of architectures")
    parser.add_argument('--exp-name', type=str, required=True, help="Name of this experiment")

    return parser.parse_args()


def merge_config_with_cli(config, args):
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
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    final_config = merge_config_with_cli(config, args)

    run_experiment(**final_config)

if __name__ == "__main__":
    main()