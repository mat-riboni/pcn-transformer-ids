import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_dataset(path):
    df = pd.read_csv(path) 
    X = df.drop(['Label', 'Attack'], axis=1) 
    y = df['Label']
    return X, y

def split_dataset(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test

def split_dataset_temporal(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test

def create_ssl_dataset(X_train, y_train, label_ratio=0.2):
    """
    important: this swaps the (label_ratio%) labels with the value -1.
    we treat the value -1 as 'No Label', then we create a mask for training,
    giving more importance to labeled data. 
    """
    indices = y_train.index    
    unlabeled_idx, labeled_idx = train_test_split(
        indices, test_size=label_ratio, stratify=y_train, random_state=42
    )        
    y_ssl = y_train.copy()    
    y_ssl.loc[unlabeled_idx] = -1.0
    
    return X_train, y_ssl

def get_labeled_only(X_train, y_ssl):
    mask = y_ssl != -1.0
    
    X_labeled = X_train[mask]
    y_labeled = y_ssl[mask]
    
    return X_labeled, y_labeled


def relabel_and_save_pcn(model, X_unlabeled, T_infer, eta_infer, output_csv="pseudo_labeled_data.csv", device='mps', batch_size=64):
    model.eval() 
    model.to(device)
    
    if not isinstance(X_unlabeled, torch.Tensor):
        X_tensor = torch.tensor(X_unlabeled, dtype=torch.float32)
        X_numpy = X_unlabeled
    else:
        X_tensor = X_unlabeled
        X_numpy = X_unlabeled.cpu().numpy()
        
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    
    for (x_batch,) in loader:
        B = x_batch.size(0)
        d_0 = model.dims[0]
        x_batch = x_batch.view(B, d_0).to(device)
        
        inputs_latents = [x_batch] + model.init_latents(B, device)
        weights = [layer.W for layer in model.layers] + [model.readout.weight]
        
        with torch.no_grad():
            for t in range(1, T_infer + 1):
                errors, gain_modulated_errors = model.compute_errors(inputs_latents)
                
                eps_L = torch.zeros_like(inputs_latents[-1])
                errors_extended = errors + [eps_L]
                
                for l in range(1, model.L + 1):
                    grad_Xl = errors_extended[l] - gain_modulated_errors[l-1] @ weights[l-1]
                    inputs_latents[l] -= eta_infer * grad_Xl
                    
            y_hat = model.readout(inputs_latents[-1])
            preds_binary = (y_hat > 0.5).float()
            
            all_predictions.append(preds_binary.cpu().numpy())
            
    pseudo_labels = np.vstack(all_predictions)
    num_features = X_numpy.shape[1]
    feature_cols = [f"feature_{i}" for i in range(num_features)]
    
    df = pd.DataFrame(X_numpy, columns=feature_cols)    
    df["pseudo_label"] = pseudo_labels.flatten()     
    df.to_csv(output_csv, index=False)    
    return df

def create_sequences(X, y, window_size=10):
    X_seq, y_seq = [], []
    X_arr = X.values if hasattr(X, 'values') else np.array(X)
    y_arr = y.values if hasattr(y, 'values') else np.array(y)
    
    for i in range(len(X_arr) - window_size):
        X_seq.append(X_arr[i : i + window_size])
        y_seq.append(y_arr[i + window_size - 1]) # L'etichetta è quella dell'ultimo record della finestra
        
    return torch.tensor(np.array(X_seq), dtype=torch.float32), torch.tensor(np.array(y_seq), dtype=torch.long)