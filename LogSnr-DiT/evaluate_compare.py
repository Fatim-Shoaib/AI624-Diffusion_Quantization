import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil
import argparse
import numpy as np
from diffusers import PixArtAlphaPipeline, DDIMScheduler
from tqdm import tqdm
from PIL import Image

# === ARGUMENTS ===
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="PixArt-alpha/PixArt-XL-2-1024-MS")

# DEFAULTS SET HERE
parser.add_argument("--checkpoint_standard", type=str, default="tdq_blockwise_fast.pt", help="Path to Standard TDQ checkpoint")
parser.add_argument("--checkpoint_logsnr", type=str, default="tdq_blockwise_logsnr.pt", help="Path to Log-SNR TDQ checkpoint")

parser.add_argument("--num_images", type=int, default=50)
parser.add_argument("--image_size", type=int, default=512)
parser.add_argument("--abits", type=int, default=8)
parser.add_argument("--wbits", type=int, default=8)
parser.add_argument("--save_dir", type=str, default="comparison_results")
args = parser.parse_args()

# Path Resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(script_dir, "..", "pixart_diffusers_512"))
if os.path.exists(MODEL_PATH):
    args.model_path = MODEL_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    import lpips
    from torchvision import transforms
except ImportError:
    print("Error: pip install lpips")
    exit(1)

# =========================================================
#  CORE QUANTIZATION LOGIC (Universal)
# =========================================================

FREQ_ENCODE_DIM = 64
TDQ_HIDDEN_DIM = 64
T_MAX = 10000

def compute_log_snr(timesteps, scheduler):
    if isinstance(timesteps, int):
        timesteps = torch.tensor([timesteps])
    elif timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)
    
    timesteps_cpu = timesteps.cpu()
    alphas_cumprod = scheduler.alphas_cumprod[timesteps_cpu]
    log_snr = torch.log(alphas_cumprod) - torch.log(1 - alphas_cumprod + 1e-8)
    
    if timesteps.is_cuda:
        log_snr = log_snr.to(timesteps.device)
    return log_snr

class QuantGlobalContext:
    _current_val = None
    _scheduler = None
    _method = "tdq" # 'tdq' or 'logsnr'

    @classmethod
    def configure(cls, method, scheduler):
        cls._method = method
        cls._scheduler = scheduler

    @classmethod
    def set_timestep(cls, t):
        if t is None:
            cls._current_val = None
            return

        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t])

        if cls._method == "logsnr":
            if cls._scheduler is None:
                raise ValueError("Scheduler not set for Log-SNR method")
            cls._current_val = compute_log_snr(t, cls._scheduler)
        else:
            # Standard TDQ
            cls._current_val = t

    @classmethod
    def get_context_value(cls):
        return cls._current_val
    
    @classmethod
    def get_method(cls):
        return cls._method

class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# --- Encoding Strategies ---
def frequency_encoding_standard(t, d=FREQ_ENCODE_DIM, t_max=T_MAX):
    # Standard: t in [0, 1000]
    if not isinstance(t, torch.Tensor):
        t = torch.tensor([t], dtype=torch.float32)
    t = t.float().view(-1, 1)
    encoding_list = []
    for i in range(d):
        denominator = t_max ** (i / d)
        freq = t / denominator
        encoding_list.append(torch.sin(freq))
        encoding_list.append(torch.cos(freq))
    return torch.cat(encoding_list, dim=-1)

def frequency_encoding_logsnr(log_snr, d=FREQ_ENCODE_DIM):
    # LogSNR: normalized to [0, 1]
    if not isinstance(log_snr, torch.Tensor):
        log_snr = torch.tensor([log_snr], dtype=torch.float32)
    log_snr = log_snr.float().view(-1, 1)
    normalized_snr = (log_snr + 10.0) / 20.0
    normalized_snr = torch.clamp(normalized_snr, 0.0, 1.0)
    encoding_list = []
    for i in range(d):
        freq = normalized_snr * (2 ** i)
        encoding_list.append(torch.sin(2 * np.pi * freq))
        encoding_list.append(torch.cos(2 * np.pi * freq))
    return torch.cat(encoding_list, dim=-1)

# --- Modules ---
class TDQModule(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = FREQ_ENCODE_DIM * 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, TDQ_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(TDQ_HIDDEN_DIM, TDQ_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(TDQ_HIDDEN_DIM, TDQ_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(TDQ_HIDDEN_DIM, 1),
            nn.Softplus()
        )
    def forward(self, t_encoded):
        return self.mlp(t_encoded)

class QuantizedLinear(nn.Module):
    def __init__(self, fp_layer, abits=8, wbits=8):
        super().__init__()
        self.abits = abits
        self.wbits = wbits
        
        self.weight = nn.Parameter(fp_layer.weight.data.clone().float())
        if fp_layer.bias is not None:
            self.bias = nn.Parameter(fp_layer.bias.data.clone().float())
        else:
            self.register_parameter('bias', None)
            
        w_max = self.weight.data.abs().amax(dim=1, keepdim=True)
        self.w_quant_max = 2**(wbits-1) - 1
        w_scale = w_max / self.w_quant_max
        self.register_buffer('w_scale', torch.clamp(w_scale, min=1e-8))

        self.tdq_module = TDQModule()
        self.a_zero_point = nn.Parameter(torch.zeros(1))
        self.a_quant_max = 2**abits - 1

    def forward(self, x):
        context_val = QuantGlobalContext.get_context_value()
        method = QuantGlobalContext.get_method()
        orig_dtype = x.dtype
        
        if context_val is None:
            return F.linear(x.float(), self.weight, self.bias).to(orig_dtype)

        x = x.float()
        context_val = context_val.to(x.device)

        # Switch Encoding based on Method
        if method == "logsnr":
            t_encoded = frequency_encoding_logsnr(context_val).to(x.device)
        else:
            t_encoded = frequency_encoding_standard(context_val).to(x.device)
        
        a_scale = self.tdq_module(t_encoded) + 1e-9
        
        view_shape = [x.shape[0]] + [1] * (x.ndim - 1)
        a_scale = a_scale.view(*view_shape)

        x_div = x / a_scale
        x_plus_z = x_div + self.a_zero_point
        x_int = StraightThroughEstimator.apply(x_plus_z)
        x_clipped = torch.clamp(x_int, 0, self.a_quant_max)
        x_quant = (x_clipped - self.a_zero_point) * a_scale

        w_int = StraightThroughEstimator.apply(self.weight / self.w_scale)
        w_clipped = torch.clamp(w_int, -self.w_quant_max, self.w_quant_max)
        w_quant = w_clipped * self.w_scale
        
        out = F.linear(x_quant, w_quant, self.bias)
        return out.to(orig_dtype)

    def load_tdq_data(self, data):
        # Universal Loader
        if 'tdq' in data:
            self.tdq_module.load_state_dict(data['tdq'])
        elif 'tdq_state_dict' in data:
            self.tdq_module.load_state_dict(data['tdq_state_dict'])
        
        if 'zp' in data:
            self.a_zero_point.data.copy_(data['zp'].to(self.a_zero_point.device))
        elif 'zero_point' in data:
            self.a_zero_point.data.copy_(data['zero_point'].to(self.a_zero_point.device))
        
        if 'w_scale' in data:
            self.w_scale.data.copy_(data['w_scale'].to(self.w_scale.device))

# =========================================================
#  MAIN LOGIC
# =========================================================

def calc_lpips(loss_fn, img_clean, img_quant, device):
    t = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    t_clean = t(img_clean).unsqueeze(0).to(device)
    t_quant = t(img_quant).unsqueeze(0).to(device)
    with torch.no_grad():
        dist = loss_fn(t_clean, t_quant)
    return dist.item()

def recursive_setattr(obj, attr, val):
    path = attr.split('.')
    parent = obj
    for step in path[:-1]:
        parent = getattr(parent, step)
    setattr(parent, path[-1], val)

def run_evaluation_pass(pipe, method_name, checkpoint_path, prompts, ref_dir, save_dir, loss_fn):
    print(f"\n--- Starting Pass: {method_name.upper()} ---")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return float('inf')

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    layers_replaced = 0
    for name, module in pipe.transformer.named_modules():
        if name in checkpoint:
            q_layer = QuantizedLinear(module, args.abits, args.wbits).to(DEVICE)
            q_layer.load_tdq_data(checkpoint[name])
            recursive_setattr(pipe.transformer, name, q_layer)
            layers_replaced += 1
    
    print(f"Replaced {layers_replaced} layers.")
    
    config_method = "logsnr" if method_name == "logsnr" else "tdq"
    QuantGlobalContext.configure(config_method, pipe.scheduler)
    
    scores = []
    
    def hook_fn(module, args, kwargs):
        t = kwargs.get('timestep', None)
        if t is None and len(args) > 2: t = args[2]
        QuantGlobalContext.set_timestep(t)
    
    handle = pipe.transformer.register_forward_pre_hook(hook_fn, with_kwargs=True)
    
    for i, prompt in enumerate(tqdm(prompts)):
        QuantGlobalContext.set_timestep(None)
        
        ref_path = os.path.join(ref_dir, f"ref_{i}.png")
        img_clean = Image.open(ref_path).convert("RGB")
        
        seed = 1000 + i
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        
        with torch.no_grad():
            img_quant = pipe(prompt, num_inference_steps=20, height=args.image_size, width=args.image_size, generator=generator).images[0]
            img_quant.save(os.path.join(save_dir, f"{method_name}_{i}.png"))
            
        score = calc_lpips(loss_fn, img_clean, img_quant, DEVICE)
        scores.append(score)
    
    handle.remove()
    return np.mean(scores)

def run():
    print(f"DEBUG: Loading Model...")
    pipe = PixArtAlphaPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16, use_safetensors=True).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    if os.path.exists(args.save_dir):
        try: shutil.rmtree(args.save_dir)
        except: pass
    os.makedirs(args.save_dir, exist_ok=True)
    
    prompts = [""] * args.num_images
    loss_fn = lpips.LPIPS(net='vgg').to(DEVICE).eval()

    # STEP 1: BASELINE
    print(f"\n{'='*40}")
    print(f"Generating Baseline Images (FP16)")
    print(f"{'='*40}")
    for i, prompt in enumerate(tqdm(prompts)):
        seed = 1000 + i
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        with torch.no_grad():
            img = pipe(prompt, num_inference_steps=20, height=args.image_size, width=args.image_size, generator=generator).images[0]
            img.save(os.path.join(args.save_dir, f"ref_{i}.png"))

    # STEP 2: STANDARD TDQ
    del pipe
    torch.cuda.empty_cache()
    print("\nReloading Pipeline for Standard TDQ...")
    pipe = PixArtAlphaPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16, use_safetensors=True).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    score_std = run_evaluation_pass(pipe, "standard", args.checkpoint_standard, prompts, args.save_dir, args.save_dir, loss_fn)

    # STEP 3: LOG-SNR TDQ
    del pipe
    torch.cuda.empty_cache()
    print("\nReloading Pipeline for Log-SNR TDQ...")
    pipe = PixArtAlphaPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16, use_safetensors=True).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    score_log = run_evaluation_pass(pipe, "logsnr", args.checkpoint_logsnr, prompts, args.save_dir, args.save_dir, loss_fn)

    # FINAL REPORT
    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS (Lower LPIPS is better)")
    print(f"{'='*60}")
    print(f"Method          | Avg LPIPS")
    print(f"----------------|----------")
    print(f"Standard TDQ    | {score_std:.5f}")
    print(f"Log-SNR TDQ     | {score_log:.5f}")
    print(f"----------------|----------")
    print(f"Images saved to: {os.path.abspath(args.save_dir)}")
    
    if score_log < score_std:
        print(f"\nWinner: Log-SNR (Better fidelity to baseline)")
    else:
        print(f"\nWinner: Standard (Better fidelity to baseline)")

if __name__ == "__main__":
    run()