import torch
import torch.nn as nn
import numpy as np

class NoiseScheduler:
    def __init__(self, num_timesteps=1000, schedule="linear", beta_start=0.00085, beta_end=0.0120):
        self.num_timesteps = num_timesteps
        
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        elif schedule == "sigmoid":
            self.betas = self._sigmoid_beta_schedule(num_timesteps, beta_start, beta_end)
        elif schedule == "quadratic":
            self.betas = self._quadratic_beta_schedule(num_timesteps)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def get_snr(self, timestep):
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep], dtype=torch.long)
            
        # Ensure correct device
        device = timestep.device
        if self.alphas_cumprod.device != device:
            self.alphas_cumprod = self.alphas_cumprod.to(device)

        t_idx = timestep.long().clamp(0, self.num_timesteps - 1)

        alpha_bar = self.alphas_cumprod[t_idx]
        snr = alpha_bar / (1 - alpha_bar + 1e-8)
        return snr

    def get_log_snr(self, timestep):
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep], dtype=torch.long)
            
        # Ensure correct device
        device = timestep.device
        if self.alphas_cumprod.device != device:
            self.alphas_cumprod = self.alphas_cumprod.to(device)
            
        alpha_bar = self.alphas_cumprod[timestep.long()]
        return torch.log(alpha_bar + 1e-8) - torch.log1p(-alpha_bar + 1e-8)
    
    def _cosine_beta_schedule(self, T, s=0.008):
        steps = torch.arange(T + 1, dtype=torch.float32) / T
        f = torch.cos((steps + s) / (1 + s) * torch.pi / 2) ** 2
        alphas = f[1:] / f[:-1]
        betas = 1 - alphas
        return betas.clamp(0, 0.999)
    
    def _sigmoid_beta_schedule(self, T, beta_start=None, beta_end=None):
        beta_start = beta_start if beta_start is not None else 0.0001
        beta_end = beta_end if beta_end is not None else 0.02
        x = torch.linspace(-6, 6, T)
        betas = torch.sigmoid(x) * (beta_end - beta_start) + beta_start
        return betas
    
    def _quadratic_beta_schedule(self, T, start=0.0001, end=0.02):
        t = torch.linspace(0, 1, T) ** 2
        return start + (end - start) * t