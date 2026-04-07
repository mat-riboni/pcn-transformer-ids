import torch
import time

def train_pcn_binary(model, data_loader, num_epochs, eta_infer, eta_learn,
T_infer, T_learn, lambda_fact=2, device='mps'):
    model.to(device).train()
    print("training started")
    start = time.time()
    for epoch in range(num_epochs):
        print(f"epoch: {epoch}")
        for x_batch, y_batch in data_loader:
            B = x_batch.size(0)
            d_0 = model.dims[0]
            x_batch = x_batch.view(B, d_0).to(device)
            y_batch = y_batch.view(B, 1).float().to(device)
            inputs_latents = [x_batch] + model.init_latents(B, device)
            weights = [layer.W for layer in model.layers] + [model.readout.weight]
            mask = (y_batch != -1.0).float()
            # INFERENCE - T_infer steps
            with torch.no_grad():
                for t in range(1, T_infer + 1):
                    errors, gain_modulated_errors = model.compute_errors(inputs_latents)
                    y_hat = model.readout(inputs_latents[-1])
                    eps_sup = (y_hat - y_batch) * mask * lambda_fact
                    eps_L = eps_sup @ weights[-1]
                    errors_extended = errors + [eps_L]
                    # Latent gradients and updates
                    for l in range(1, model.L + 1):
                        grad_Xl = errors_extended[l] - \
                        gain_modulated_errors[l-1] @ weights[l-1]
                        inputs_latents[l] -= eta_infer * grad_Xl
            # LEARNING - T_learn steps
            with torch.no_grad():
                for t in range(T_infer + 1, T_learn + T_infer + 1):
                    errors, gain_modulated_errors = model.compute_errors(inputs_latents)
                    y_hat = model.readout(inputs_latents[-1])
                    eps_sup = (y_hat - y_batch) * mask * lambda_fact
                    # Weight gradients and updates
                    for l in range(model.L):
                        grad_Wl = -(gain_modulated_errors[l].T @ inputs_latents[l+1]) / B
                        weights[l] -= eta_learn * grad_Wl
                    grad_Wout = eps_sup.T @ inputs_latents[-1] / B
                    weights[-1] -= eta_learn * grad_Wout
        end_epoch_time = time.time()
        elapsed_time = end_epoch_time - start
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        time_remaining = avg_time_per_epoch * (num_epochs - epoch - 1)
        print(f"Time remaining: {time_remaining//60}m at {time.time()}")
