from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import numpy as np
from tqdm import tqdm



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





# (Assumendo che get_device sia definito altrove)

def evaluate_pcn_binary(model, test_loader, T_infer, eta_infer, threshold=0.4):
    device = torch.device('mps') # o get_device()
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


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_pcn_anomaly(model, test_loader, T_infer, eta_infer, threshold_energy=23, device='mps'):
    model.eval()
    model.to(device)

    all_energies = []
    all_labels = []

    # Energies calc
    for x_batch, y_batch in tqdm(test_loader):
        B = x_batch.size(0)
        x_batch = x_batch.view(B, model.dims[0]).to(device)
        y_batch = y_batch.view(B).to(device)

        latents = model.init_latents(B, device)

        for t in range(T_infer):
            energy = model.compute_energy(latents, x_batch)
            mean_energy = energy.mean()
            mean_energy.backward()

            with torch.no_grad():
                for z in latents:
                    z -= eta_infer * z.grad
                    z.grad = None

        with torch.no_grad():
            final_energy = model.compute_energy(latents, x_batch)

        all_energies.extend(final_energy.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    all_energies = np.array(all_energies)
    all_labels = np.array(all_labels)

    plt.figure(figsize=(12, 6))

    normal_energies = all_energies[all_labels == 0.0]
    attack_energies = all_energies[all_labels == 1.0]

    max_plot_val = np.percentile(all_energies, 99) 
    bins = np.linspace(0, max_plot_val, 100)

    # Creiamo i due istogrammi. 
    # alpha=0.5 li rende semi-trasparenti per vedere le sovrapposizioni.
    # density=True normalizza l'asse Y (utile se il dataset è molto sbilanciato)
    plt.hist(normal_energies, bins=bins, alpha=0.6, label='Normal (Class 0)', color='#1f77b4', density=True)
    plt.hist(attack_energies, bins=bins, alpha=0.6, label='Attack (Class 1)', color='#d62728', density=True)

    # Aggiungiamo una linea verticale che indica la soglia che hai scelto
    plt.axvline(x=threshold_energy, color='black', linestyle='dashed', linewidth=2, label=f'Threshold ({threshold_energy})')

    plt.title("Energy distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Energy", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

    preds = (all_energies > threshold_energy).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, preds))

    return all_energies, all_labels