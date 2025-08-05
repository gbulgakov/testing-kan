# Bridging KANs and Tabular DL: 

This repository contains an experimental pipeline for exploring Kolmogorov-Arnold Networks (KAN) applied to Tabular Deep Learning (DL). KANs offer a promising alternative to traditional neural networks by leveraging spline-based representations for better interpretability and performance on structured data.

## Overview

Kolmogorov-Arnold Networks (KANs) are inspired by the Kolmogorov-Arnold representation theorem, providing a novel architecture for modeling complex functions. This article delves into their application on tabular data, comparing them with standard deep learning models like MLPs, and highlighting advantages in accuracy, efficiency, and explainability.

Key highlights from the article:
- **Theoretical Foundations**: Explanation of KAN architecture and how it differs from MLPs.
- **Use of modern Tabular DL approaches benefit KAN performance.
- **Experiments on Tabular Datasets**: Benchmarks on popular datasets.

If you're interested in advanced DL techniques for tabular data, this repo is a great starting point!

## Quickstart

To run experiments from the paper:

1. **Clone the Repository**:
```
git clone https://anonymous.4open.science/r/tabular-kan
cd testing-kan
```

2. **Install Dependencies** (assuming Python 3.8+):
Install the reqiurenments using conda environment:
```
conda enc create -f env.yaml
conda activate testing-kan
```
After following this two steps, you can run python `scripts/run.py` specifying the following arguments:

1. --model_names = list of model names separated by spaces of model you want to run. This implementation contain these models: small_kan (base kan implementation), kan (using increased tuning space), fast_kan, cheby_kan (ChebyshevKAN), mlp
2. --dataset_names = list of dataset names. (churn, california, house, adult, diamond, otto, higgs-small, black-friday, fb-comments)
3. --emb_names = list of embedding strategies (none, PLE-Q, periodic)
4. --arch_types = list of ensembling strategies (plain, tabm-mini, tabm)
5. --config = path to the .yaml config file, that contains additional info of the experiment.

Alternetivly, you can specify the arguments above in the config file, and only use --config while running ` python scripts/run.py`


