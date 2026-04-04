from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch

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

import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def evaluate_pcn_binary(model, test_loader, T_infer, eta_infer, threshold=0.4):
    device = get_device()
    model.eval()
    model.to(device)
    
    all_preds = []
    all_trues = []
    
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
    
    print(f"Accuracy: {acc:.4f}\n")
    print(conf_matrix)
    print(class_report)
    
    return acc, conf_matrix, class_report