import torch
import torch.nn as nn
import numpy as np

class FourierEmbedding(nn.Module):
    def __init__(self, dim=64, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        # Enforce 1D input to avoid [N,1,1] broadcasting issues
        t = t.view(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding
    
class TDQ_MLP(nn.Module):
    def __init__(self, time_dim=64, hidden_dim=64):
        super().__init__()
        self.embed = FourierEmbedding(dim=time_dim)
        
        self.net = nn.Sequential(
            nn.Linear(time_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1), 
            nn.Softplus() 
        )
    def forward(self, t):
        t_emb = self.embed(t)
        interval = self.net(t_emb)
        return interval
    
class SNR_TDQ_MLP(nn.Module):
    def __init__(self, hidden_dim=64, embedding_dim=64):
        super().__init__()
        self.embed = FourierEmbedding(dim=embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 128),    
            nn.SiLU(),            
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Softplus() 
        )

    def forward(self, log_snr):
        # Enforce view(-1) is handled inside embed, but good to be explicit
        x_emb = self.embed(log_snr) 
        interval = self.net(x_emb)
        return interval