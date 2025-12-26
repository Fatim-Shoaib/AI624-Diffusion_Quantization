import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import argparse
import numpy as np
import copy
import gc
from diffusers import PixArtAlphaPipeline, DDIMScheduler
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="PixArt-alpha/PixArt-XL-2-512x512")
parser.add_argument("--abits", type=int, default=8)
parser.add_argument("--wbits", type=int, default=8)
parser.add_argument("--calib_samples", type=int, default=16)
parser.add_argument("--timesteps_per_sample", type=int, default=20)
parser.add_argument("--opt_steps", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--save_path", type=str, default="tdq_blockwise_logsnr.pt")
parser.add_argument("--image_size", type=int, default=512)
parser.add_argument("--block_step", type=int, default=4, help="Optimize every Nth block. 1=All, 2=Every other, etc.")
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(script_dir, "..", "pixart_diffusers_512"))
if os.path.exists(MODEL_PATH):
    args.model_path = MODEL_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FREQ_ENCODE_DIM = 64
TDQ_HIDDEN_DIM = 64

# === LOG-SNR CALCULATION ===
def compute_log_snr(timesteps, scheduler):
    """
    Compute log-SNR from timesteps using the scheduler's alpha values.
    log-SNR = log(alpha_t^2 / (1 - alpha_t^2))
    """
    if isinstance(timesteps, int):
        timesteps = torch.tensor([timesteps])
    elif timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)
    
    # Move timesteps to CPU to match alphas_cumprod device
    timesteps_cpu = timesteps.cpu()
    
    # Get alpha_cumprod values for these timesteps
    alphas_cumprod = scheduler.alphas_cumprod[timesteps_cpu]
    
    # Compute log-SNR: log(alpha^2 / (1 - alpha^2))
    log_snr = torch.log(alphas_cumprod) - torch.log(1 - alphas_cumprod + 1e-8)
    
    # Move back to original device if needed
    if timesteps.is_cuda:
        log_snr = log_snr.to(timesteps.device)
    
    return log_snr

# === GLOBAL CONTEXT ===
class QuantGlobalContext:
    _current_log_snr = None
    _scheduler = None

    @classmethod
    def set_scheduler(cls, scheduler):
        cls._scheduler = scheduler

    @classmethod
    def set_timestep(cls, t):
        """Convert timestep to log-SNR and store"""
        if t is None:
            cls._current_log_snr = None
            return
            
        if cls._scheduler is None:
            cls._current_log_snr = None
            return
            
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t])
            
        cls._current_log_snr = compute_log_snr(t, cls._scheduler)

    @classmethod
    def get_log_snr(cls):
        return cls._current_log_snr

# === UTILS ===
class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def frequency_encoding(log_snr, d=FREQ_ENCODE_DIM):
    """
    Frequency encoding for log-SNR values.
    Adapts the paper's frequency encoding to work with log-SNR range.
    """
    if not isinstance(log_snr, torch.Tensor):
        log_snr = torch.tensor([log_snr], dtype=torch.float32)
    
    log_snr = log_snr.float().view(-1, 1)
    
    # Normalize log-SNR to a reasonable range for frequency encoding
    # Typical log-SNR range is [-10, 10], we map this to [0, 1]
    normalized_snr = (log_snr + 10.0) / 20.0
    normalized_snr = torch.clamp(normalized_snr, 0.0, 1.0)
    
    encoding_list = []
    for i in range(d):
        # Use exponential frequency scaling
        freq = normalized_snr * (2 ** i)
        encoding_list.append(torch.sin(2 * np.pi * freq))
        encoding_list.append(torch.cos(2 * np.pi * freq))
    
    return torch.cat(encoding_list, dim=-1)

# === MODULES ===
class TDQModule(nn.Module):
    """Time-Dependent Quantization MLP for log-SNR conditioning"""
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
        
    def forward(self, log_snr_encoded):
        return self.mlp(log_snr_encoded)

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
        log_snr = QuantGlobalContext.get_log_snr()
        orig_dtype = x.dtype
        
        if log_snr is None:
            return F.linear(x.float(), self.weight, self.bias).to(orig_dtype)

        x = x.float()
        log_snr = log_snr.to(x.device)

        # 1. Dynamic Scale Generation via TDQ
        log_snr_encoded = frequency_encoding(log_snr).to(x.device)
        a_scale = self.tdq_module(log_snr_encoded) + 1e-9

        view_shape = [x.shape[0]] + [1] * (x.ndim - 1)
        a_scale = a_scale.view(*view_shape)

        # 2. Asymmetric Activation Quantization
        x_div = x / a_scale
        x_plus_z = x_div + self.a_zero_point
        x_int = StraightThroughEstimator.apply(x_plus_z)
        x_clipped = torch.clamp(x_int, 0, self.a_quant_max)
        x_quant = (x_clipped - self.a_zero_point) * a_scale

        # 3. Symmetric Weight Quantization
        w_int = StraightThroughEstimator.apply(self.weight / self.w_scale)
        w_clipped = torch.clamp(w_int, -self.w_quant_max, self.w_quant_max)
        w_quant = w_clipped * self.w_scale
        
        out = F.linear(x_quant, w_quant, self.bias)
        return out.to(orig_dtype)

    def initialize_tdq(self, activation_stats):
        with torch.no_grad():
            flat = activation_stats.flatten().float()
            g_min, g_max = flat.min(), flat.max()
            
            init_scale = (g_max - g_min) / self.a_quant_max
            init_scale = torch.clamp(init_scale, min=1e-5)
            
            init_zp = torch.round(-g_min / init_scale)
            self.a_zero_point.data.fill_(init_zp)

            last_linear = list(self.tdq_module.mlp.modules())[-2]
            nn.init.kaiming_normal_(last_linear.weight)
            
            val = torch.exp(init_scale) - 1
            if val > 0:
                init_bias = torch.log(val).item()
            else:
                init_bias = init_scale.item()
            last_linear.bias.data.fill_(init_bias)

# === DATA CONTAINER ===
class BlockInputCache:
    def __init__(self):
        self.args = []   
        self.kwargs = [] 
        self.log_snrs = []  # Changed from timesteps to log_snrs

# === FAST OPTIMIZATION LOGIC ===
def collect_initial_inputs(pipe, samples, steps_per_sample):
    cache = BlockInputCache()
    first_block = pipe.transformer.transformer_blocks[0]
    
    def hook_fn(module, args, kwargs, output):
        safe_args = []
        for a in args:
            safe_args.append(a.detach() if isinstance(a, torch.Tensor) else a)
        
        safe_kwargs = {}
        for k, v in kwargs.items():
            safe_kwargs[k] = v.detach() if isinstance(v, torch.Tensor) else v
            
        log_snr = QuantGlobalContext.get_log_snr()
        
        cache.args.append(tuple(safe_args))
        cache.kwargs.append(safe_kwargs)
        cache.log_snrs.append(log_snr.detach())
        raise StopIteration("Input Collected")

    orig_forward = first_block.forward
    def wrapped_forward(*args, **kwargs):
        hook_fn(first_block, args, kwargs, None)
        return orig_forward(*args, **kwargs)
    
    first_block.forward = wrapped_forward
    
    print("  > Collecting initial inputs (Fast Mode with Log-SNR)...")
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    latent_dim = args.image_size // 8
    
    with torch.no_grad():
        enc = pipe.encode_prompt("", device=DEVICE, do_classifier_free_guidance=False)
        prompt_embeds, prompt_attn = enc[0], enc[1]
        
        total = samples * steps_per_sample
        pbar = tqdm(total=total, desc="Collecting")
        
        for _ in range(samples):
            lat = torch.randn((1, 4, latent_dim, latent_dim), generator=generator, device=DEVICE, dtype=torch.float16)
            ts = torch.randint(0, 1000, (steps_per_sample,), generator=generator, device=DEVICE)
            
            for t in ts:
                QuantGlobalContext.set_timestep(t)
                alpha = pipe.scheduler.alphas_cumprod[t.item()]
                noisy_lat = (alpha**0.5)*lat + ((1-alpha)**0.5)*torch.randn_like(lat)
                
                added_cond = {
                    "resolution": torch.tensor([args.image_size, args.image_size], device=DEVICE).unsqueeze(0),
                    "aspect_ratio": torch.tensor([1.0], device=DEVICE).unsqueeze(0)
                }
                
                try:
                    pipe.transformer(
                        noisy_lat,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attn,
                        timestep=t.unsqueeze(0),
                        added_cond_kwargs=added_cond
                    )
                except StopIteration:
                    pass
                except Exception as e:
                    print(f"Error: {e}")
                    
                pbar.update(1)

    first_block.forward = orig_forward
    QuantGlobalContext.set_timestep(None)
    
    # Print log-SNR statistics
    all_log_snrs = torch.stack(cache.log_snrs)
    print(f"\n  Log-SNR Statistics:")
    print(f"    Min: {all_log_snrs.min():.4f}")
    print(f"    Max: {all_log_snrs.max():.4f}")
    print(f"    Mean: {all_log_snrs.mean():.4f}")
    print(f"    Std: {all_log_snrs.std():.4f}\n")
    
    return cache

def pass_through_block(block, cache):
    """
    Runs the block in FP16 mode without optimization/quantization.
    Used for skipped blocks.
    """
    print("  > Passing through skipped block (FP16)...")
    next_cache = BlockInputCache()
    block.eval()
    
    with torch.no_grad():
        for i in range(len(cache.args)):
            c_args = cache.args[i]
            c_kwargs = cache.kwargs[i]
            c_log_snr = cache.log_snrs[i]
            
            QuantGlobalContext._current_log_snr = c_log_snr
            
            b_args = tuple(a.to(DEVICE) if isinstance(a, torch.Tensor) else a for a in c_args)
            b_kwargs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in c_kwargs.items()}
            
            # Forward FP16
            out = block(*b_args, **b_kwargs)
            if isinstance(out, tuple): out = out[0]
            
            # Update cache for next block
            new_hidden = out.detach()
            old_args = cache.args[i]
            new_args = (new_hidden,) + old_args[1:] # Replace hidden_states
            
            next_cache.args.append(new_args)
            next_cache.kwargs.append(cache.kwargs[i])
            next_cache.log_snrs.append(cache.log_snrs[i])
            
    return next_cache

def optimize_block_fast(block, cache):
    # 1. Generate Targets
    print("  > Generating Targets...")
    targets = []
    block.eval()
    block.to(DEVICE)
    
    with torch.no_grad():
        for i in range(len(cache.args)):
            c_args = cache.args[i]
            c_kwargs = cache.kwargs[i]
            c_log_snr = cache.log_snrs[i]
            
            QuantGlobalContext._current_log_snr = c_log_snr
            
            b_args = tuple(a.to(DEVICE) if isinstance(a, torch.Tensor) else a for a in c_args)
            b_kwargs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in c_kwargs.items()}
            
            out = block(*b_args, **b_kwargs)
            if isinstance(out, tuple): out = out[0]
            targets.append(out.detach())

    # 2. Replace & Optimize
    print("  > Optimizing Layers...")
    layers_to_opt = []
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            q_layer = QuantizedLinear(module, args.abits, args.wbits).to(DEVICE)
            w_abs = module.weight.abs().max()
            q_layer.initialize_tdq(torch.tensor([-w_abs, w_abs], device=DEVICE))
            
            parent_name = name.rpartition('.')[0]
            if parent_name:
                parent = block.get_submodule(parent_name)
                setattr(parent, name.split('.')[-1], q_layer)
            
            layers_to_opt.append(q_layer)
            
    if not layers_to_opt:
        return pass_through_block(block, cache)

    params_mlp = []
    params_zp = []
    for ql in layers_to_opt:
        params_mlp.extend(ql.tdq_module.parameters())
        params_zp.append(ql.a_zero_point)
        
    optimizer = optim.AdamW([
        {'params': params_mlp, 'lr': 1e-4}, 
        {'params': params_zp, 'lr': 1e-3}
    ])
    
    block.train()
    num_data = len(cache.args)
    indices = torch.randperm(num_data).tolist()
    
    pbar = tqdm(range(args.opt_steps), desc="Optimizing", leave=False)
    data_idx = 0
    
    for _ in pbar:
        optimizer.zero_grad()
        batch_loss = 0
        
        for _ in range(args.batch_size):
            idx = indices[data_idx % num_data]
            data_idx += 1
            
            c_args = cache.args[idx]
            c_kwargs = cache.kwargs[idx]
            c_log_snr = cache.log_snrs[idx]
            c_target = targets[idx].to(DEVICE)
            
            QuantGlobalContext._current_log_snr = c_log_snr
            
            b_args = tuple(a.to(DEVICE) if isinstance(a, torch.Tensor) else a for a in c_args)
            b_kwargs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in c_kwargs.items()}
            
            q_out = block(*b_args, **b_kwargs)
            if isinstance(q_out, tuple): q_out = q_out[0]
            
            loss = F.mse_loss(q_out.float(), c_target.float())
            loss.backward()
            batch_loss += loss.item()
            
        optimizer.step()
        pbar.set_description(f"Loss: {batch_loss/args.batch_size:.5f}")
        
    block.eval()

    # 3. Generate Inputs for Next Block (Using Quantized Output)
    print("  > Generating next inputs...")
    next_cache = BlockInputCache()
    
    with torch.no_grad():
        for i in range(len(cache.args)):
            c_args = cache.args[i]
            c_kwargs = cache.kwargs[i]
            c_log_snr = cache.log_snrs[i]
            
            QuantGlobalContext._current_log_snr = c_log_snr
            
            b_args = tuple(a.to(DEVICE) if isinstance(a, torch.Tensor) else a for a in c_args)
            b_kwargs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in c_kwargs.items()}
            
            q_out = block(*b_args, **b_kwargs)
            if isinstance(q_out, tuple): q_out = q_out[0]
            
            new_hidden = q_out.detach()
            old_args = cache.args[i]
            new_args = (new_hidden,) + old_args[1:]
            
            next_cache.args.append(new_args)
            next_cache.kwargs.append(cache.kwargs[i])
            next_cache.log_snrs.append(cache.log_snrs[i])

    del targets
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    return next_cache

def run():
    print(f"Loading Model: {args.model_path}")
    pipe = PixArtAlphaPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16, use_safetensors=True).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # Set scheduler for log-SNR computation
    QuantGlobalContext.set_scheduler(pipe.scheduler)
    
    blocks = pipe.transformer.transformer_blocks
    total_blocks = len(blocks)
    
    print(f"Found {total_blocks} Transformer Blocks.")
    print(f"Optimization Strategy: Optimize 1, Skip {args.block_step - 1} (Step={args.block_step})")
    print(f"Using Log-SNR for TDQ conditioning\n")
    
    print("Step 1: Collecting Initial Inputs...")
    current_cache = collect_initial_inputs(pipe, args.calib_samples, args.timesteps_per_sample)
    
    print("Step 2: Sequential Optimization...")
    for i in range(total_blocks):
        if i % args.block_step == 0:
            print(f"\n[{i+1}/{total_blocks}] OPTIMIZING Block {i}...")
            current_cache = optimize_block_fast(blocks[i], current_cache)
        else:
            print(f"\n[{i+1}/{total_blocks}] SKIPPING Block {i} (Pass-through)...")
            current_cache = pass_through_block(blocks[i], current_cache)
        
    print("\nSaving Checkpoint...")
    state_dict = {}
    quantized_count = 0
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, QuantizedLinear):
            state_dict[name] = {
                'tdq_state_dict': module.tdq_module.state_dict(),
                'zero_point': module.a_zero_point.cpu(),
                'w_scale': module.w_scale.cpu()
            }
            quantized_count += 1
            
    torch.save(state_dict, args.save_path)
    print(f"Saved {quantized_count} quantized layers to {args.save_path}")

    print("\nGenerating Test Image...")
    def inference_hook(module, args, kwargs):
        t = kwargs.get('timestep', None)
        if t is None and len(args) > 2: t = args[2]
        QuantGlobalContext.set_timestep(t)
    
    handle = pipe.transformer.register_forward_pre_hook(inference_hook, with_kwargs=True)
    
    with torch.no_grad():
        img = pipe("a cyberpunk cat in tokyo, neon lights, 4k", num_inference_steps=20).images[0]
        img.save("result_blockwise_logsnr.png")
    
    handle.remove()
    print("Done.")

if __name__ == "__main__":
    run()