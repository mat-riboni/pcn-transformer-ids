
import torch
import torch.nn as nn
from PCN_layer import PCNLayer

class PredictiveCodingNetwork(nn.Module):
    def __init__(self,
    dims,
    output_dim
    ):
        super().__init__()
        self.dims = dims
        self.L = len(dims) - 1
        self.layers = nn.ModuleList([
        PCNLayer(in_dim=dims[l+1],
        out_dim=dims[l])
        for l in range(self.L)
        ])
        self.readout = nn.Linear(dims[-1], output_dim, bias=False)
    def init_latents(self, batch_size, device):
        return [
        torch.randn(batch_size, d, device=device,
        requires_grad=False)
        for d in self.dims[1:]
        ]
    def compute_errors(self, inputs_latents):
        errors, gain_modulated_errors = [], []
        for l, layer in enumerate(self.layers):
            x_hat, a = layer(inputs_latents[l + 1])
            err = inputs_latents[l] - x_hat
            gm_err = err * layer.activation_deriv(a)
            errors.append(err)
            gain_modulated_errors.append(gm_err)
        return errors, gain_modulated_errors