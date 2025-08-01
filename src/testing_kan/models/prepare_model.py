import torch.nn as nn
import torch
from typing import Optional
from torch import Tensor
import torch.nn.functional as F
import rtdl_num_embeddings
from .efficient_kan import KAN
from .fastkan import FastKAN
from .mlp import MLP
from .chebyshev_kan import ChebyKAN

# prepare model
def model_init_preparation(config, dataset, model_name, arch_type, emb_name):
    dataset_info = dataset['info']
    # num_cont_cols = dataset['train']['X_num'].shape[1]
    # num_cat_cols = 0
    # if dataset_info['n_cat_features'] > 0:
    #     num_cat_cols = dataset['train']['X_cat'].shape[1]
    num_cont_cols = dataset_info['num_cont_cols']
    num_cat_cols = dataset_info['num_cat_cols']
    num_classes = 1
    if dataset_info['task_type'] == 'multiclass':
        num_classes = dataset_info['n_classes']
        

    # building model
    in_features = num_cont_cols * config.get('d_embedding', 1) + num_cat_cols
    out_features = num_classes
    backbone = None
    layer_widths = None
    layer_kwargs = {}
 
    if model_name == 'kan' or model_name == 'small_kan':
        layer_widths = [in_features] + [config['kan_width'] for i in range(config['kan_layers'])] + [out_features]
        layer_kwargs = {
            'grid_size' : config['grid_size']
        }
        backbone = KAN(layer_widths, batch_norm=False, **layer_kwargs)
    
    elif model_name == 'batch_norm_kan':
        layer_widths = [in_features] + [config['kan_width'] for i in range(config['kan_layers'])] + [out_features]
        layer_kwargs = {
            'grid_size' : config['grid_size']
        }
        backbone = KAN(layer_widths, batch_norm=True, **layer_kwargs)

    elif model_name == 'fast_kan':
        layer_widths = [in_features] + [config['kan_width'] for i in range(config['kan_layers'])] + [out_features]
        layer_kwargs = {
            'num_grids' : config['grid_size']
        }
        backbone = FastKAN(layer_widths, **layer_kwargs)

    elif model_name == 'cheby_kan':
        layer_widths = [in_features] + [config['kan_width'] for i in range(config['kan_layers'])] + [out_features]
        layer_kwargs = {
            'degree' : config['degree']
        }
        backbone = ChebyKAN(layers_hidden=layer_widths, **layer_kwargs)

    elif model_name == 'mlp':
        layer_widths = [in_features] + [config['mlp_width'] for i in range(config['mlp_layers'])] + [out_features]
        layer_kwargs = {}
        dropout = config['dropout']
        backbone = MLP(layer_widths, dropout)
        
        
    # MLP and KAN connected sequentially with shared layer width
    elif model_name == 'mlp_kan':
        mlp_layer_widths = [in_features] + [config['mlp_width'] for i in range(config['mlp_layers'])] + [config['kan_width']]
        kan_layer_widths = [config['kan_width']] + [config['kan_width'] for i in range(config['kan_layers'])] + [out_features]
        dropout = config['dropout']
        backbone = nn.Sequential(
            MLP(mlp_layer_widths, dropout),
            KAN(kan_layer_widths, grid_size=config['grid_size'], batch_norm=False)
        )
        layer_widths = mlp_layer_widths + kan_layer_widths
    
    elif model_name == 'kan_mlp':
        kan_layer_widths = [in_features] + [config['kan_width'] for i in range(config['kan_layers'])] + [config['kan_width']]
        mlp_layer_widths = [config['kan_width']] + [config['mlp_width'] for i in range(config['mlp_layers'])] + [out_features]
        dropout = config['dropout']
        backbone = nn.Sequential(
            KAN(kan_layer_widths, grid_size=config['grid_size'], batch_norm=False),
            MLP(mlp_layer_widths, dropout)
        )
        layer_widths = kan_layer_widths + mlp_layer_widths


    # some embeddings requiere all data
    if emb_name not in ['none', 'periodic']:
        X_num = dataset['train']['X_num']
        Y = dataset['train']['y']
    # building embeddings
    if emb_name == 'piecewiselinearq' or emb_name == 'PLE-Q':
        bins = rtdl_num_embeddings.compute_bins(X=X_num, n_bins=config['d_embedding'])
        num_embeddings = {
            'type': 'PiecewiseLinearEmbeddings',
            'd_embedding': config['d_embedding'],
            'activation': False,
            'version': 'B'
        }
    elif emb_name == 'piecewiselineart' or emb_name == 'PLE-T': # was not used
        tree_kwargs = {'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4}
        bins = rtdl_num_embeddings.compute_bins(X=X_num, y=Y, n_bins=config['d_embedding'], regression=True, tree_kwargs=tree_kwargs)
        num_embeddings = {
            'type': 'PiecewiseLinearEmbeddings',
            'd_embedding': config['d_embedding'],
            'activation': False ,
            'version': 'B'
        }
    else:
        bins = None
        if emb_name == 'periodic' or emb_name == 'PLR':
            num_embeddings = {
                'type': 'PeriodicEmbeddings',
                'd_embedding': config['d_embedding'],
                'lite': True,
                'frequency_init_scale': config['sigma'],
                'n_features': num_cont_cols
            }
        elif emb_name == 'kan_emb':
            num_embeddings = {
                'type': '_NKANLinear',
                'in_features': 1,
                'out_features': config['d_embedding'],
                'grid_size' : config['emb_grid_size'],
                'n': num_cont_cols
            }
        elif emb_name == 'fast_kan_emb':
            num_embeddings = {
                'type': '_NFastKANLayer',
                'input_dim': 1,
                'output_dim': config['d_embedding'],
                'num_grids' : config['emb_grid_size'],
                'n': num_cont_cols
                # 'd_embedding': config['d_embedding']
            }
        else:
            num_embeddings = None
            
    task_type = dataset_info['task_type']
    loss_fn = None
    
    if task_type == 'binclass':
        loss_fn = F.binary_cross_entropy_with_logits
    elif task_type == 'multiclass':
        loss_fn = F.cross_entropy
    else:
        loss_fn =  F.mse_loss
    k = None
    if arch_type != 'plain':
        k = 16
        
    return layer_widths, layer_kwargs, backbone, bins, num_embeddings, loss_fn, k
    
