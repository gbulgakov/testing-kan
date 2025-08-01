import zipfile
import os
import gc
import yaml
import torch
from .optimizers.ademamix import AdEMAMix
from .optimizers.muon import Muon
from .optimizers.mars import MARS
from .optimizers.sign import Signum

OPTIMIZERS = { 
              'adamw' : torch.optim.AdamW,
              'ademamix' : AdEMAMix,
              'mars' : MARS,
              'muon' : Muon,
              'sign' : Signum,
              'momentum_sign' : Signum
             }

def create_zip_archive(source_dir, archive_path):
    """
    Collects  data from source_dir to one Zip-archive.
    """
    #  'w' for new archive and ZIP_DEFLATED for compression
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir(source_dir):
            # full path to added file
            file_path = os.path.join(source_dir, filename)
            # check whether it is file
            if os.path.isfile(file_path) and not filename.endswith('.zip'):
                # adding to archive. arcname=filename ensures,
                # that there would be not extra pathes in archive (only file name)
                zipf.write(file_path, arcname=filename)



def seed_everything(seed=0):
    import random
    random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def get_optimizer(optim_name, model_params, config):
    optim_class = OPTIMIZERS[optim_name]
    optim_kwargs = {'lr' : config['lr']}

    if optim_name != 'muon':             # for muon all params are  -- muon_params
        optim_kwargs['weight_decay'] = config['weight_decay']
    if optim_name == 'momentum_sign':    # basic signum - without momentum
        optim_kwargs['momentum'] = 0.9

    if optim_name == 'muon':
        model_params = list(model_params)
    return optim_class(model_params, **optim_kwargs)
 
def clean_up_model(model, optimizer):
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

