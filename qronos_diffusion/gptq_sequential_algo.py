import math
import torch
import torch.nn as nn

def quantize_weight(x, scale, zero, maxq):
    if maxq < 0: return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class GPTQSequentialAlgo:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.columns = layer.weight.data.shape[1]
        
        # GPTQ only uses H = X^T X
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float64)
        self.nsamples = 0

    def add_batch(self, inp):
        # Flatten [Batch, Seq, Dim] -> [Samples, Dim]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            if inp.shape[0] == 1: inp = inp.squeeze(0)
            
        inp = inp.t().to(torch.float64) # [Dim, Samples]
        
        current_samples = inp.shape[1]
        
        # Accumulate H
        if self.nsamples == 0:
            self.H = inp.matmul(inp.t())
        else:
            self.H *= self.nsamples / (self.nsamples + current_samples)
            scaler = math.sqrt(2 / self.nsamples) if self.nsamples > 0 else 1.0
            inp = scaler * inp
            self.H += inp.matmul(inp.t())
            
        self.nsamples += current_samples

    def fasterquant(self, quantizer, blocksize=128, percdamp=0.01, groupsize=-1):
        W = self.layer.weight.data.clone().to(dtype=torch.float64)
        
        if not quantizer.ready():
            quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Dampening
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        
        # Cholesky Inverse (Standard GPTQ)
        try:
            L_full = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L_full)
            # GPTQ typically uses the upper triangular of the inverse for the loop
            Hinv = torch.linalg.cholesky(Hinv, upper=True) 
        except RuntimeError:
            print("Hessian not positive definite, skipping this layer.")
            return

        Q = torch.zeros_like(W)
        Losses = torch.zeros_like(W)

        # Main Loop
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        quantizer.find_params(W[:, (i1 + i):min((i1 + i + groupsize), self.columns)], weight=True)

                q = quantize_weight(w.unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq).flatten()
                Q1[:, i] = q
                
                # GPTQ Error Calculation
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            
            # Global update
            if i2 < self.columns:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)