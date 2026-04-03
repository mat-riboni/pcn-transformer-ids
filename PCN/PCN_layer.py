import torch
from torch import nn
from torch import autocast

class PCNLayer(nn.Module):
    def __init__(self,
        in_dim,                                         #layer above dimension
        out_dim,                                        #current layer dimension
        activation_fn=torch.relu,
        activation_deriv=lambda a: (a > 0).float()      #derivate of layer below (linear because of ReLU)
    ):  
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        torch.nn.init.xavier_uniform_(self.W)
        self.activation_fn = activation_fn
        self.activation_deriv = activation_deriv

    def forward(self, x_above):
        with autocast(device_type='cuda'):
            a = x_above @ self.W.T
            x_hat = self.activation_fn(a)
            return x_hat, a