import torch
import torch.nn as nn
import torch.optim as optim

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train_baseline_transformer(model, train_loader, test_loader=None, epochs=10, lr=0.0003, weight_decay=0.0005):
    device = get_device()
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        val_msg = ""
        if test_loader is not None:
            model.eval() 
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            val_acc = correct / total
            history['val_acc'].append(val_acc)
            val_msg = f" | Validation Accuracy: {val_acc:.4f}"
            
        print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {avg_train_loss:.4f}{val_msg}")
        
    return model, history

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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