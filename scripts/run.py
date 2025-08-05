from __future__ import annotations

import argparse
import os
from typing import Dict, Any, List

import torch
import yaml

# Limit NumPy/OpenBLAS thread usage for reproducibility
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from run_experiment import run_experiment


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment with config and CLI override")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config")
    parser.add_argument('--datasets', nargs='+', help="List of datasets names")
    parser.add_argument('--models', nargs='+', help="List of model names")
    parser.add_argument('--embs', nargs='+', help="List of embeddings")
    parser.add_argument('--optim-names', nargs='+', help="List of optimizers")
    parser.add_argument('--arch-types', nargs='+', help="List of architectures")
    parser.add_argument('--exp-name', type=str, required=True, help="Name of this experiment")
    return parser.parse_args()


def _merge_config_with_cli(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge YAML config with CLI overrides."""
    merged = config.copy()
    merged["exp_name"] = args.exp_name
    
    if args.datasets:
        merged["dataset_names"] = args.datasets
    if args.models:
        merged["model_names"] = args.models
    if args.embs:
        merged["emb_names"] = args.embs
    if args.optim_names:
        merged["optim_names"] = args.optim_names
    if args.arch_types:
        merged["arch_types"] = args.arch_types

    return merged


def main():
    args = _parse_cli()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    final_config = _merge_config_with_cli(config, args)
    run_experiment(**final_config)

if __name__ == "__main__":
    main()