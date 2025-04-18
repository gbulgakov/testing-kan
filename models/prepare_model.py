import torch.nn as nn
import torch
from typing import Optional
from torch import Tensor
import torch.nn.functional as F
import rtdl_num_embeddings
from models.efficient_kan import KAN
from models.fastkan import FastKAN

# обычная MLP
class MLP(nn.Sequential):
    def __init__(self, layers, dropout):
        super(MLP, self).__init__()
        
        total_layers = []
        for n_in, n_out in zip(layers[:-2], layers[1:-1]):
            total_layers.append(nn.Linear(n_in, n_out))
            total_layers.append(nn.SiLU(inplace=False))
            total_layers.append(nn.Dropout(dropout, inplace=False))
        total_layers.append(nn.Linear(layers[-2], layers[-1])) # выходной слой

        self.classifier = nn.Sequential(*total_layers)

# модель с эмбеддингами
class ModelWithEmbedding(nn.Module):
    def __init__(
        self,
        n_cont_features,
        d_embedding,
        emb_name,
        backbone_model,
        bins, sigma=None # словарь всех необязательных параметров, например sigma, bins
    ) -> None:
        super().__init__()
        self.d_embedding = d_embedding
        self.emb_name = emb_name
        
        if emb_name == 'periodic':
            self.cont_embeddings = rtdl_num_embeddings.PeriodicEmbeddings(
                n_cont_features, d_embedding, frequency_init_scale=sigma, lite=True
            )
            
        if emb_name == 'piecewiselinearq' or emb_name == 'piecewiselineart' or emb_name == 'PLE-Q' or emb_name == 'PLE-T':
            self.cont_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                d_embedding=d_embedding, activation=False, version='B', bins=bins
            )

        self.backbone = backbone_model
    
    def forward(
        self,
        x_num : Tensor,
        x_cat : Optional[Tensor] = None
    ) -> Tensor:
        x = []
        # Step 1. Embed the continuous features.
        # Flattening is needed for MLP-like models.
        if self.emb_name != 'none':
              x_num = self.cont_embeddings(x_num)
        x.append(x_num.flatten(1))
        
        #categorical features do not need embeddings
        if x_cat is not None:
            x.append(x_cat.flatten(1))
        
        x = torch.column_stack(x)
        return self.backbone(x)

# подготовка модели 
def model_init_preparation(config, dataset, model_name, emb_name):
    dataset_info = dataset['info']
    num_cont_cols = dataset['train']['X_num'].shape[1]
    num_cat_cols = 0
    if dataset_info['n_cat_features'] > 0:
        num_cat_cols = dataset['train']['X_cat'].shape[1]
    num_classes = 1
    if dataset_info['task_type'] == 'multiclass':
        num_classes = dataset_info['n_classes']
        

    # создание модели

    in_features = num_cont_cols * config.get('d_embedding', 1) + num_cat_cols
    out_features = num_classes
    backbone = None

    if model_name == 'kan' or model_name == 'small_kan':
        layer_widths = [in_features] + [config['kan_width'] for i in range(config['kan_layers'])] + [out_features]
        backbone = KAN(layer_widths, grid_size=config['grid_size'], batch_norm=True)
    
    elif model_name == 'fast_kan':
        layer_widths = [in_features] + [config['kan_width'] for i in range(config['kan_layers'])] + [out_features]
        backbone = FastKAN(layer_widths, num_grids=config['grid_size'])

    elif model_name == 'mlp':
        layer_widths = [in_features] + [config['mlp_width'] for i in range(config['mlp_layers'])] + [out_features]
        dropout = (config['dropout'] if config['use_dropout'] else 0)
        backbone = MLP(layer_widths, dropout)
        
    # ниже MLP и KAN соединены последовательно с шириной первого/последнего KAN слоя
    elif model_name == 'mlp_kan':
        mlp_layer_widths = [in_features] + [config['mlp_width'] for i in range(config['mlp_layers'])] + [config['kan_width']]
        kan_layer_widths = [config['kan_width']] + [config['kan_width'] for i in range(config['kan_layers'])] + [out_features]
        dropout = (config['dropout'] if config['use_dropout'] else 0)
        backbone = nn.Sequential(
            MLP(mlp_layer_widths, dropout),
            KAN(kan_layer_widths, grid_size=config['grid_size'], batch_norm=True)
        )
        layer_widths = mlp_layer_widths + kan_layer_widths
    
    elif model_name == 'kan_mlp':
        kan_layer_widths = [in_features] + [config['kan_width'] for i in range(config['kan_layers'])] + [config['kan_width']]
        mlp_layer_widths = [config['kan_width']] + [config['mlp_width'] for i in range(config['mlp_layers'])] + [out_features]
        dropout = (config['dropout'] if config['use_dropout'] else 0)
        backbone = nn.Sequential(
            KAN(kan_layer_widths, grid_size=config['grid_size'], batch_norm=True),
            MLP(mlp_layer_widths, dropout)
        )
        layer_widths = kan_layer_widths + mlp_layer_widths
    
    # создание эмбеддингов
    if emb_name == 'piecewiselinearq' or emb_name == 'PLE-Q':
        bins = rtdl_num_embeddings.compute_bins(dataset['train']['X_num'], n_bins=config['d_embedding'])
    elif emb_name == 'piecewiselineart' or emb_name == 'PLE-T': # это мы  больше не используем
        tree_kwargs = {'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4} #возможно стоит тюнить
        bins = rtdl_num_embeddings.compute_bins(X=dataset['train']['X_num'], y=dataset['train']['y'], n_bins=config['d_embedding'], regression=True, tree_kwargs=tree_kwargs)
    else:
        bins = None
            
    task_type = dataset_info['task_type']
    loss_fn = None
    
    if task_type == 'binclass':
        loss_fn = F.binary_cross_entropy_with_logits
    elif task_type == 'multiclass':
        loss_fn = F.cross_entropy
    else:
        loss_fn =  F.mse_loss
        
    return layer_widths, backbone, bins, loss_fn
    
