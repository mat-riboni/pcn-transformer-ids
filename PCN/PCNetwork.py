
import torch
import torch.nn as nn
import torch.nn.functional as F
from .PCN_layer import PCNLayer

class PredictiveCodingNetwork(nn.Module):
    def __init__(self,
    dims,
    ):
        super().__init__()
        self.dims = dims
        self.L = len(dims) - 1
        self.layers = nn.ModuleList([
            PCNLayer(in_dim=dims[l+1], out_dim=dims[l], activation_fn=nn.GELU())
            for l in range(self.L)
        ])

    def init_latents(self, batch_size, device, prev_latents=None, dynamic_decay=None):
        if prev_latents is None:
            return [
                torch.zeros(batch_size, d, device=device, requires_grad=True)
                for d in self.dims[1:]
            ]
        else:
            if dynamic_decay is None:
                dynamic_decay = torch.full((batch_size, 1), 0.9, device=device)
            
            return [
                (z.detach() * dynamic_decay).clone().requires_grad_(True)
                for z in prev_latents
            ]
    
    def compute_energy(self, inputs_latents, x_batch):
        latents_with_input = [x_batch] + inputs_latents 
        
        batch_energy = torch.zeros(x_batch.size(0), device=x_batch.device)
        
        for l, layer in enumerate(self.layers):
            x_below = latents_with_input[l]       
            x_above = latents_with_input[l + 1]   
            
            x_hat = layer(x_above)
            
            mse_per_feature = (x_hat - x_below) ** 2
            layer_energy = mse_per_feature.sum(dim=1) 
            
            batch_energy += layer_energy
            
        return batch_energy



