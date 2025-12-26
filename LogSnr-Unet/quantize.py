import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantGlobalContext:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QuantGlobalContext, cls).__new__(cls)
            cls._instance.current_log_snr = None
            cls._instance.error_history = []
            cls._instance.measure_error = True
        return cls._instance

    def set_current_snr(self, log_snr):
        self.current_log_snr = log_snr

    def get_current_snr(self):
        return self.current_log_snr

    def add_error(self, mse_value):
        if self.measure_error:
            self.error_history.append(mse_value)

    def get_and_reset_error_stats(self):
        if not self.error_history:
            return 0.0, 0.0
        avg_error = sum(self.error_history) / len(self.error_history)
        max_error = max(self.error_history)
        self.error_history = []
        return avg_error, max_error

class QLayer(nn.Module):
    def __init__(self, original_module, mlp, abits=8, wbits=8):
        super().__init__()
        self.module = original_module 
        self.net = mlp                
        self.abits = abits
        self.wbits = wbits
        self.context = QuantGlobalContext() 
        self.quantize_act = True
        self.quantize_w = True

    def forward(self, x, *args, **kwargs):
        if x.dim() <= 1 or (x.dim() == 2 and x.shape[1] == 1):
            return self.module(x, *args, **kwargs)

        if self.quantize_act:
            log_snr = self.context.get_current_snr()
            if log_snr is not None and log_snr.device != x.device:
                log_snr = log_snr.to(x.device)
            S = self.net(log_snr)
            x_int = self.quantize_activations(x, S)
            x_fake = self.dequantize(x_int, S)
        else:
            x_fake = x

        if self.quantize_w:
            w_int, w_s = self.quantize_weights(self.module.weight)
            w_fake = self.dequantize(w_int, w_s)
        else:
            w_fake = self.module.weight
        
        y_quant = F.linear(x_fake, w_fake, self.module.bias)

        if self.context.measure_error:
            with torch.no_grad():
                y_ref = F.linear(x, self.module.weight, self.module.bias)
                mse = F.mse_loss(y_quant, y_ref)
                self.context.add_error(mse.item())

        return y_quant

    def quantize_weights(self, weight):
        qmax = 2 ** (self.wbits - 1) - 1
        scale = weight.abs().max() / qmax
        scale = torch.max(scale, torch.tensor(1e-8, device=weight.device, dtype=weight.dtype))
        wint = torch.round(weight / scale).clamp(-qmax, qmax)
        return wint, scale

    def quantize_activations(self, x, S):
        qmax = 2 ** (self.abits - 1) - 1
        qmin = -(2 ** (self.abits - 1))
        
        S = S.to(x.device).to(x.dtype)
        S = torch.max(S, torch.tensor(1e-5, device=x.device, dtype=x.dtype))
        
        xint = torch.round(x / S).clamp(qmin, qmax)
        return xint

    def dequantize(self, xint, scale):
        scale = scale.to(xint.device).to(xint.dtype)
        xfake = xint * scale
        return xfake

    def load_quant_params(self, path):
        self.net.load_state_dict(torch.load(path))

    def save_quant_params(self, path):
        torch.save(self.net.state_dict(), path)

    def extra_repr(self):
        return f"w_bits={self.wbits}, a_bits={self.abits}"