import torch.nn as nn

class TransformerBaseline(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=4, num_layers=4, ff_dim=256, dropout=0.2):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            batch_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, x):
        
        x = self.embedding(x)        
        x = self.transformer(x)
        
        x_last_step = x[:, -1, :] 
        
        out = self.classifier(x_last_step)
        return out