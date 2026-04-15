import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=2, min_delta=0, path='best_pcn_model.pth', min_epochs=5):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.min_epochs = min_epochs

    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  
            if epoch > self.min_epochs:
                self.save_checkpoint(model)

        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        print(f"Weights saved in: {self.path}")