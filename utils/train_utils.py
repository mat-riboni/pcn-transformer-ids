from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math




def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def evaluate_model(model, test_loader):
    device = get_device()
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds)
    
    print(f"Accuracy: {acc:.4f}\n")
    print(conf_matrix)
    print(class_report)
    
    return acc, conf_matrix, class_report






def evaluate_pcn_binary(model, test_loader, T_infer, eta_infer, threshold=0.4):
    device = torch.device(get_device) # o get_device()
    model.eval()
    model.to(device)
    
    all_preds = []
    all_trues = []
    
    total_abs_error = 0.0
    total_error_elements = 0
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            B = x_batch.size(0)
            d_0 = model.dims[0]
            
            x_batch = x_batch.view(B, d_0).to(device)
            y_batch = y_batch.to(device) 
            
            inputs_latents = [x_batch] + model.init_latents(B, device)
            weights = [layer.W for layer in model.layers] + [model.readout.weight]
            
            for t in range(1, T_infer + 1):
                errors, gain_modulated_errors = model.compute_errors(inputs_latents)
                
                eps_L = torch.zeros_like(inputs_latents[-1])
                errors_extended = errors + [eps_L]
                
                if t == T_infer:
                    batch_error_sum = sum(torch.abs(e).sum().item() for e in errors_extended)
                    batch_error_elements = sum(e.numel() for e in errors_extended)
                    
                    total_abs_error += batch_error_sum
                    total_error_elements += batch_error_elements
                
                for l in range(1, model.L + 1):
                    grad_Xl = errors_extended[l] - gain_modulated_errors[l-1] @ weights[l-1]
                    inputs_latents[l] -= eta_infer * grad_Xl
                    
            y_hat = model.readout(inputs_latents[-1])
            
            preds_binary = (y_hat > threshold).float()
            
            all_preds.extend(preds_binary.cpu().numpy())
            all_trues.extend(y_batch.cpu().numpy())
            
    all_preds = np.array(all_preds).flatten()
    all_trues = np.array(all_trues).flatten()
    
    acc = accuracy_score(all_trues, all_preds)
    conf_matrix = confusion_matrix(all_trues, all_preds)
    class_report = classification_report(all_trues, all_preds)
    
    mean_final_abs_error = total_abs_error / total_error_elements if total_error_elements > 0 else 0
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Mean Absolute Error (t={T_infer}): {mean_final_abs_error:.6f}\n")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    
    return acc, conf_matrix, class_report, mean_final_abs_error



def evaluate_pcn_anomaly(model, test_loader, T_infer, eta_infer, threshold_energy=23, device='mps', save_img=False, plot_name='energy_density.png'):
    model.eval()
    model.to(device)

    all_energies = []
    all_labels = []

    prev_latents = None
    prev_energy = None


    for x_batch, y_batch in tqdm(test_loader):
        B = x_batch.size(0)
        x_batch = x_batch.view(B, model.dims[0]).to(device)
        y_batch = y_batch.view(B).to(device)

        if prev_latents is not None and prev_latents[0].size(0) != B:
                prev_latents = None
                prev_energy = None

        if prev_energy is not None:
            current_decay = calculate_energy_based_decay(prev_energy)
        else:
            current_decay = None 

        latents = model.init_latents(B, device, prev_latents=prev_latents, dynamic_decay=current_decay)

        for t in range(T_infer):
            energy = model.compute_energy(latents, x_batch)
            mean_energy = energy.mean()
            mean_energy.backward()

            with torch.no_grad():
                for z in latents:
                    grad_clipped = torch.clamp(z.grad, min=-1.0, max=1.0)
                    z -= eta_infer * grad_clipped 
                    z.grad = None

        with torch.no_grad():
            final_energy = model.compute_energy(latents, x_batch)

        
        prev_latents = [z.detach() for z in latents]
        prev_energy = final_energy.detach()
        all_energies.extend(final_energy.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    all_energies = np.array(all_energies)
    all_labels = np.array(all_labels)

    plt.figure(figsize=(12, 6))

    normal_energies = all_energies[all_labels == 0.0]
    attack_energies = all_energies[all_labels == 1.0]

    max_plot_val = np.percentile(all_energies, 99) 
    bins = np.linspace(0, max_plot_val, 100)

    plt.hist(normal_energies, bins=bins, alpha=0.6, label='Normal (Class 0)', color='#1f77b4', density=True)
    plt.hist(attack_energies, bins=bins, alpha=0.6, label='Attack (Class 1)', color='#d62728', density=True)

    plt.axvline(x=threshold_energy, color='black', linestyle='dashed', linewidth=2, label=f'Threshold ({threshold_energy})')

    plt.title("Energy distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Energy", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if not save_img:
        plt.show()
    else:
        plt.savefig(plot_name, dpi=400, bbox_inches='tight')
        plt.close()

    preds = (all_energies > threshold_energy).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, preds))

    return all_energies, all_labels

def calculate_adaptive_decay(time_since_anomaly, gamma_base=0.50, gamma_max=0.99, k_decay=0.2):
    """
    Calcola il fattore di decadimento della memoria latente (State-Dependent Forgetting).
    
    Args:
        time_since_anomaly (int): Numero di batch trascorsi dall'ultimo evento ad alta energia.
        gamma_base (float): Decadimento asintotico (oblio rapido per la normalità).
        gamma_max (float): Decadimento iniziale (massima ritenzione al momento dello shock).
        k_decay (float): Velocità di rilassamento (più è alto, più in fretta scende a gamma_base).
        
    Returns:
        float: Il fattore di decadimento calcolato per il batch corrente.
    """
    return gamma_base + (gamma_max - gamma_base) * math.exp(-k_decay * time_since_anomaly)

import torch

def calculate_energy_based_decay(energy, gamma_base=0.95, gamma_shock=0.1, shock_scale=20.0):
    """
    Calcola il fattore di decadimento basato sull'energia istantanea.
    
    Args:
        energy (torch.Tensor): Vettore delle energie dell'ultimo step (forma: [B]).
        gamma_base (float): Decadimento normale quando l'energia è bassa (alta ritenzione).
        gamma_shock (float): Decadimento in caso di shock (bassa ritenzione, si resetta la memoria).
        shock_scale (float): Parametro che controlla quanto velocemente il decadimento
                             passa da base a shock all'aumentare dell'energia.
                             
    Returns:
        torch.Tensor: Vettore dei decadimenti per il batch corrente (forma: [B, 1]).
    """
    # Usiamo una sigmoide scalata per avere una transizione morbida
    # Alta energia -> esponente negativo grande -> exp va a 0 -> decay = gamma_shock
    # Bassa energia -> esponente vicino a 0 -> exp va a 1 -> decay = gamma_base
    
    # Normalizziamo l'energia per renderla indipendente dalla scala assoluta (opzionale ma consigliato)
    # Esempio: energy_norm = energy / (energy.mean() + 1e-8) 
    
    decay_factor = gamma_shock + (gamma_base - gamma_shock) * torch.exp(-energy / shock_scale)
    return decay_factor.unsqueeze(1) # Ritorna [B, 1] per poter moltiplicare le latenti [B, dim]