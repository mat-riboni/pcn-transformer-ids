import torch
from torch import nn
from torch import autocast

class PCNLayer(nn.Module):
    def __init__(self,
        in_dim,                                         #layer above dimension
        out_dim,                                        #current layer dimension
        activation_fn=nn.GELU,
    ):  
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation_fn = activation_fn

    def forward(self, x_above):
        a = self.linear(x_above)
        x_hat = self.activation_fn(a)
        return x_hat