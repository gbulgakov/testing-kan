import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def reshape(x, n):
  return einops.rearrange(x, 'b (c n) ... -> b c n ...', n=n)

def reshape_back(x):
  return einops.rearrange(x, 'b c n ... -> b (c n) ...')

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, output_channels=256, height=16, width=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_channels * height * width)  # 256 * 16 * 1 = 4096
        self.output_channels = output_channels
        self.height = height
        self.width = width

    def forward(self, x):
        x = self.linear(x)  # (batch_size, 4096)
        x = einops.rearrange(x, 'b (c h w) -> b c h w', c=self.output_channels, h=self.height, w=self.width)
        return x  # (batch_size, 256, 16, 1)


class ModReLU(nn.Module):

    def __init__(self, n, ch, norm="gn"):
        super().__init__()
        self.n = n
        if norm == "bn":
            self.norm = nn.BatchNorm2d(ch // n)
        elif norm == "gn":
            self.norm = nn.GroupNorm(ch // n, ch // n)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = reshape(x, self.n)
        m = torch.linalg.norm(x, dim=2)  # [B C//n H, W]
        m = torch.nn.ReLU()(self.norm(m))  # No relu
        x = m.unsqueeze(2) * F.normalize(x, dim=2)
        x = reshape_back(x)
        return x

class KConv2d(nn.Module):

  def __init__(self, n, ch, connectivity='conv', ksize=3, init_omg=1.0, hw=(16,16), use_omega=True, use_omega_c=True):
    # connnectivity is either 'conv' or 'ca'
    super().__init__()
    assert (ch % n) == 0
    self.n = n
    self.ch = ch

    if connectivity == 'conv':
      self.connectivity = nn.Conv2d(ch, ch, kernel_size=(ksize, 1), stride=1, padding=(ksize//2, 0), bias=False)
    elif connectivity == 'conv_mlp':
        self.connectivity = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=(ksize, 1), stride=1, padding=(ksize//2, 0), bias=False),
            ModReLU(n, ch),
            nn.Conv2d(ch, ch, kernel_size=(ksize, 1), stride=1, padding=(ksize//2, 0), bias=False))
    else:
      raise NotImplementedError

    self.use_omega = use_omega
    self.use_omega_c = use_omega_c
    if use_omega or use_omega_c:
      if n == 2 :
        self.omg_param = nn.Parameter(torch.randn(ch//2, 2))
      else:
        self.omg_param = nn.Parameter(init_omg * (1/np.sqrt(n))* torch.randn(ch//n, n, n))

  def omg(self, p):
    if self.n==2:
      p = torch.linalg.norm(p, dim=1)
      return torch.stack(
          [torch.stack([torch.zeros_like(p), p], -1),
          torch.stack([-p, torch.zeros_like(p)], -1)],
              -1)
    else:
      return p - p.transpose(1, 2)

  def forward(self, x, c=None):
    y = self.connectivity(x)
    if c is not None:
      y = y + c
    y = reshape(y, self.n)
    x = reshape(x, self.n)

    omg_x = torch.einsum('cnm,bcmhw->bcnhw', self.omg(self.omg_param), x) if self.use_omega else torch.zeros_like(x)
    proj = y - torch.sum(y*x, 2, keepdim=True) * x
    if c is not None:
      c = reshape(c, self.n)
      omg_c = torch.einsum('cnm,bcmhw->bcnhw', self.omg(self.omg_param), c) if self.use_omega_c else torch.zeros_like(c)
      return reshape_back(omg_x + proj), reshape_back(omg_c)
    else:
      return reshape_back(omg_x + proj)

  def compute_energy(self, x, c=None):

    y = self.connectivity(x)
    y = y + c
    B = x.shape[0]
    return - torch.sum(x.view(B, -1) * y.view(B, -1), -1)


class KBlock(nn.Module):

  def __init__(self, n, ch, connectivity='conv', T=4, ksize=7, init_omg=0.1, c_norm='gn', use_omega=True, use_omega_c=True):
    super().__init__()
    self.n = n
    self.ch = ch
    self.T = T
    self.kconv = KConv2d(n, ch, connectivity=connectivity, ksize=ksize, init_omg=init_omg, use_omega=use_omega, use_omega_c=use_omega_c)
    self.monitor_count = 0
    if c_norm == 'gn':
      self.c_norm = nn.GroupNorm(ch//n, ch)
    else:
      self.c_norm = lambda x: x

  def normalize(self, x, y=None):
    x = reshape(x, self.n)
    x = torch.nn.functional.normalize(x, dim=2)
    if y is not None:
      x = torch.linalg.norm(reshape(y, self.n), dim=2, keepdim=True) * x
    x = reshape_back(x)
    return x

  def monitor_norms(self, dt, c, x):

    def print_norm(x, name):
      x = x.view(x.shape[0], -1).detach()
      x = torch.linalg.norm(x, dim=1).mean(0)
      print(f"avg norms of {name}: {x:.6f}")
    for x, name in ((dt, 'dt'), (c, 'c'), (x, 'x')):
      print_norm(x, name)

  def forward(self, x, c, T, gamma, del_t=1.0, return_xs=False, return_es=False, T_noc=None):
    x = self.normalize(x)
    c = self.c_norm(c)
    xs = [x]
    es = []
    if return_es:
      energy = self.kconv.compute_energy(x, c)
      es.append(energy)

    if self.monitor_count >= 50:
      do_monitoring = False
      self.monitor_count = 0
    else:
      do_monitoring = False
      self.monitor_count += 1

    for t in range(T):
      dxdt, dcdt = self.kconv(x, c)
      _c = c + gamma*del_t*dcdt
      c = self.normalize(_c, c)
      x = x + gamma*del_t*dxdt
      x = self.normalize(x)

      if return_es:
        energy = self.kconv.compute_energy(x, c)
        es.append(energy)

      if return_xs:
        xs.append(x)

    if return_es:
      return x, xs, es
    else:
      return x, xs

class KNet(nn.Module):
    def __init__(self,  
                 in_features, 
                 out_features,
                 num_layers,
                 n=4, 
                 ch=32,
                 emb_len=16,
                 connectivity='conv', 
                 T=4, 
                 ksize=3, 
                 init_omg=0.1, 
                 c_norm='gn', 
                 use_omega=True, 
                 use_omega_c=True
    ):
        super().__init__()

        # Encoder to 2D format
        self.encoder = FeatureEncoder(in_features, output_channels=ch, height=emb_len)

        # Conv layers list
        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_layers):
           self.conv_layers.append(KBlock(n, ch, connectivity, T, ksize, init_omg, c_norm, use_omega, use_omega_c))

        # FC classifier or regressor
        self.fc_layer = nn.Linear(in_features=ch * emb_len, out_features=out_features)
        

    def forward(self, x):
       x = self.encoder(x)
       for conv_layer in self.conv_layers:
          x = conv_layer(x)
       x = torch.flatten(x, start_dim=1)
       x = self.fc_layer(x)
       return x
        