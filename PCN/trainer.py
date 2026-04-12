import torch
import time
from tqdm import tqdm
import torch.nn.functional as F

def train_pcn_binary(model, data_loader, num_epochs,  eta_infer, eta_learn,
T_infer, margin_attack = 50, device='mps'):
    
    model.to(device).train()
    optimizer_weights = torch.optim.AdamW(model.parameters(), lr=eta_learn)

    print("training started")
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for x_batch, y_batch in tqdm(data_loader, mininterval=20.0):
            B = x_batch.size(0)
            d_0 = model.dims[0]

            x_batch = x_batch.view(B, d_0).to(device)
            y_batch = y_batch.view(B).float().to(device)

            latents = model.init_latents(B, device)

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
                        z -= eta_infer * z.grad  
                        z.grad = None
            
            # Weights learn
            for p in model.parameters():
                p.requires_grad = True

            optimizer_weights.zero_grad()
            final_latents = [z.detach() for z in latents]
            final_energy = model.compute_energy(final_latents, x_batch)
            loss_normal = (final_energy * mask_normal).mean()
            loss_unlabeled = (final_energy * mask_unlabeled).mean()
            loss_attack = (F.relu(margin_attack - final_energy) * mask_attack).mean()
            total_loss = loss_normal + loss_unlabeled + loss_attack

            total_loss.backward()
            optimizer_weights.step()

            epoch_loss += total_loss.item()
        print(f"Epoch: {epoch + 1} | Loss: {epoch_loss/len(data_loader):.4f} ")   
            