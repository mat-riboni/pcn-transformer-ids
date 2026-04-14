import torch
import time
from tqdm import tqdm
import torch.nn.functional as F

def train_pcn_binary(model, data_loader, num_epochs,  eta_infer, eta_learn,
T_infer, margin_attack = 200, device='mps'):
    
    model.to(device).train()
    optimizer_weights = torch.optim.AdamW(model.parameters(), lr=eta_learn)

    print("training started")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_loss_norm = 0.0
        epoch_loss_atk = 0.0
        epoch_loss_unl = 0.0

        prev_latents = None
        prev_energy = None

        for x_batch, y_batch in tqdm(data_loader):
            B = x_batch.size(0)
            d_0 = model.dims[0]

            x_batch = x_batch.view(B, d_0).to(device)
            y_batch = y_batch.view(B).float().to(device)

            if prev_latents is not None and prev_latents[0].size(0) != B:
                prev_latents = None
                prev_energy = None

            if prev_energy is not None:
                current_decay = calculate_energy_based_decay(prev_energy)
            else:
                current_decay = None 

            latents = model.init_latents(B, device, prev_latents=prev_latents, dynamic_decay=current_decay)
            mask_unlabeled = (y_batch == -1.0).float()
            mask_normal    = (y_batch == 0.0).float()
            mask_attack    = (y_batch == 1.0).float()

            for p in model.parameters():
                p.requires_grad = False
            
            # INFERENCE 
            for t in range(T_infer):

                energy = model.compute_energy(latents, x_batch)
                mean_energy = energy.mean()
                mean_energy.backward()

                with torch.no_grad():
                    for z in latents:
                        grad_clipped = torch.clamp(z.grad, min=-1.0, max=1.0)
                        z -= eta_infer * grad_clipped  
                        z.grad = None
            
            # Weights learn
            for p in model.parameters():
                p.requires_grad = True

            optimizer_weights.zero_grad()
            prev_latents = [z.detach() for z in latents]
            final_latents = [z.detach().requires_grad_(True) for z in latents]
            final_energy = model.compute_energy(final_latents, x_batch)
            prev_energy = final_energy.detach()

                
            loss_normal = (final_energy * mask_normal).mean()
            loss_unlabeled = (final_energy * mask_unlabeled).mean()
            loss_attack = ((F.relu(margin_attack - final_energy) * mask_attack).mean()) * 0.1
            total_loss = loss_normal + loss_unlabeled + loss_attack

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_weights.step()

            epoch_loss += total_loss.item()
            epoch_loss_norm += loss_normal.item()
            epoch_loss_atk += loss_attack.item()
            epoch_loss_unl = loss_unlabeled.item()
            num_batches = len(data_loader)
        print(f"Epoch: {epoch + 1} | Tot: {epoch_loss/num_batches:.2f} | Norm: {epoch_loss_norm/num_batches:.2f} | Atk: {loss_attack.item()/num_batches:.2f}")

import math

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
            