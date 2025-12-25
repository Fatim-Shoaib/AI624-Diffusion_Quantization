import math
import time
import torch
import torch.nn as nn
from act_quant import dynamic_quantize_activations

def quantize_weight(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class QronosW4A4:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.columns = layer.weight.data.shape[1]
        
        # Qronos W4A4 Matrices (Float64 for precision)
        # H_quant = X_quant^T * X_quant (Curvature of the quantized input view)
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float64)
        
        # G = X_quant^T * X_real (Correction term)
        self.G = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float64)
        self.nsamples = 0

    def add_batch(self, inp):
        # 1. Preprocess Input
        # Flatten [Batch, Seq, Dim] -> [Samples, Dim]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            if inp.shape[0] == 1: inp = inp.squeeze(0)
        
        # 2. Generate Quantized Activations (Simulate W4A4 input)
        # We assume the model will quantize activations per-token during inference
        # Input shape here is [Samples, Dim]
        inp_quant = dynamic_quantize_activations(inp, bits=4)
        
        # 3. Prepare for Matrix Multiplication [Dim, Samples]
        inp = inp.t().to(torch.float64)
        inp_quant = inp_quant.t().to(torch.float64)
        
        # 4. Accumulate
        current_samples = inp.shape[1]
        
        # Running Average Update
        self.H *= self.nsamples / (self.nsamples + current_samples)
        self.G *= self.nsamples / (self.nsamples + current_samples)
        self.nsamples += current_samples
        
        # Scaling factor
        scaler = math.sqrt(2 / self.nsamples)
        inp = scaler * inp
        inp_quant = scaler * inp_quant
        
        # STRICT QRONOS W4A4 MATH:
        # H is based on QUANTIZED inputs (since that's what the quantized weights will see)
        self.H += inp_quant.matmul(inp_quant.t())
        
        # G connects QUANTIZED inputs to REAL targets
        self.G += inp_quant.matmul(inp.t()) 

    def fasterquant(self, quantizer, blocksize=128, percdamp=0.01, groupsize=-1, beta=1e4):
        W = self.layer.weight.data.clone().to(dtype=torch.float64)
        W_orig = W.clone()
        
        if not quantizer.ready():
            quantizer.find_params(W, weight=True)

        H = self.H
        G = self.G
        del self.H, self.G
        
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        G[dead, dead] = 1
        W[:, dead] = 0
        W_orig[:, dead] = 0

        # Dampening
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        
        # Initial Inversion
        try:
            L_full = torch.linalg.cholesky(H)
            iH = torch.cholesky_inverse(L_full)
        except RuntimeError:
            print("Warning: H not positive definite. Adding extra dampening.")
            H[diag, diag] += damp * 10
            L_full = torch.linalg.cholesky(H)
            iH = torch.cholesky_inverse(L_full)

        # --- QRONOS STEP 1: Global Correction (Correcting the Past) ---
        Dh = torch.diag(H)
        Dhi = torch.where(Dh != 0, 1.0 / Dh, torch.zeros_like(Dh)) 
        Uh = torch.triu(H, 1)

        # Correct the first weight column
        # Formula: w_new = (G_0 * D_inv) * w_orig - (U_0 * D_inv) * w_current
        # Note: G[:, 0] is the correlation of the 0th input feature with all others
        Gw = W_orig.matmul(G[:, 0] * Dhi[0]) 
        Uv = W.matmul(Uh[0, :] * Dhi[0])
        W[:, 0] = Gw - Uv
        
        # SMW Update for Inverse Hessian
        c = iH[0, 0]
        b = iH[1:, 0].unsqueeze(1)
        iH_sub = iH[1:, 1:] - (b.matmul(b.t()) / c)
        
        # --- QRONOS STEP 2: Error Diffusion (Shaping the Future) ---
        try:
            L = torch.linalg.cholesky(iH_sub * beta, upper=True) / math.sqrt(beta)
        except RuntimeError:
            L = torch.linalg.cholesky(iH[1:,1:] * beta, upper=True) / math.sqrt(beta)
        Hinv = L 

        Q = torch.zeros_like(W)
        
        # Quantize index 0 (Explicitly)
        Q[:, 0] = quantize_weight(W[:, 0].unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq).flatten()
        
        # Main Loop (Start from 1)
        for i1 in range(1, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            
            hinv_start = i1 - 1
            hinv_end = i2 - 1
            
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[hinv_start:hinv_end, hinv_start:hinv_end]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        quantizer.find_params(W[:, (i1 + i):min((i1 + i + groupsize), self.columns)], weight=True)

                q = quantize_weight(
                    w.unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq
                ).flatten()
                
                Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            
            if i2 < self.columns:
                remaining_hinv = Hinv[hinv_start:hinv_end, hinv_end:]
                W[:, i2:] -= Err1.matmul(remaining_hinv)

        torch.cuda.synchronize()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)