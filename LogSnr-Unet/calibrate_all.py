import torch
import torch.nn as nn
import torch.optim as optim
import gc
import os
import json
import numpy as np
from diffusers import StableDiffusionPipeline, DDIMScheduler
# Ensure these imports match your file structure
from networks import SNR_TDQ_MLP, TDQ_MLP 
from scheduler import NoiseScheduler

# --- Configuration ---
MODEL_ID = "runwayml/stable-diffusion-v1-5" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QUANT_METHOD = "tdq" 

SAVE_PATH = f"legacy/quant_checkpoint_{QUANT_METHOD}.pt"
DEBUG_JSON_PATH = f"legacy/debug_stats_{QUANT_METHOD}.json"

NUM_CALIB_IMAGES = 16
NUM_INFERENCE_STEPS = 50
CALIBRATION_BATCH_SIZE = 2 
SKIP_PATTERNS = ["time_emb", "time_proj", "class_embedding"]

def get_all_linear_layers(module, parent_name=""):
    layers = []
    for name, child in module.named_children():
        next_name = f"{parent_name}.{name}" if parent_name else name
        if isinstance(child, nn.Linear):
            if not any(s in next_name for s in SKIP_PATTERNS):
                layers.append(next_name)
        else:
            layers.extend(get_all_linear_layers(child, next_name))
    return layers

def get_module_by_name(root, dotted_path):
    cur = root
    for p in dotted_path.split('.'):
        if p.isdigit(): cur = cur[int(p)]
        else: cur = getattr(cur, p)
    return cur

def get_robust_scale(activation):
    flattened = activation.abs().reshape(-1)
    numel = flattened.numel()
    if numel == 0: return torch.tensor(1.0, device=activation.device)
    k = int(numel * 0.999)
    k_val, _ = torch.kthvalue(flattened, k)
    return ((k_val / 127.0) * 1.05).detach()

def initialize_mlp_bias(mlp, target_mean):
    last_linear = [m for m in mlp.modules() if isinstance(m, nn.Linear)][-1]
    target_val = max(target_mean.item(), 1e-6)
    try:
        init_bias = float(np.log(np.exp(target_val) - 1))
        if np.isnan(init_bias) or np.isinf(init_bias): init_bias = 0.0
    except: init_bias = 0.0
    with torch.no_grad():
        last_linear.weight.fill_(0.0)
        last_linear.bias.fill_(init_bias)

class OnlineStatsCollector:
    def __init__(self, layer_names):
        self.stats = {ln: [] for ln in layer_names}
    def get_hook(self, layer_name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.stats[layer_name].append(get_robust_scale(output).cpu().item())
        return hook

def calibrate_and_save():
    print(f"--- Starting Calibration for {QUANT_METHOD} ---")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS)

    all_linear_layers = get_all_linear_layers(pipe.unet)
    collector = OnlineStatsCollector(all_linear_layers)
    handles = [get_module_by_name(pipe.unet, ln).register_forward_hook(collector.get_hook(ln)) 
               for ln in all_linear_layers]

    print(f"Collecting activation stats on {NUM_CALIB_IMAGES} images...")
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    for i in range(0, NUM_CALIB_IMAGES, CALIBRATION_BATCH_SIZE):
        with torch.no_grad():
            pipe(prompt=[""] * CALIBRATION_BATCH_SIZE, num_inference_steps=NUM_INFERENCE_STEPS, generator=generator)
        gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()

    for h in handles: h.remove()

    # Prepare inputs for the MLP training
    timesteps_cpu = pipe.scheduler.timesteps.cpu() # 0 to 1000
    noise_sched = NoiseScheduler(num_timesteps=1000, schedule="linear")
    
    if QUANT_METHOD == "snr-tdq":
        log_snr = noise_sched.get_log_snr(timesteps_cpu)
        snr_mean, snr_std = log_snr.mean(), log_snr.std()
        # Normalize SNR
        model_input = ((log_snr - snr_mean) / (snr_std + 1e-6)).view(-1, 1).to(DEVICE)
        mlp_class = SNR_TDQ_MLP
    else: 
        # Normalize time (0-1000 -> 0.0-1.0)
        model_input = (timesteps_cpu.float() / 1000.0).view(-1, 1).to(DEVICE)
        snr_mean, snr_std = 0.0, 1.0 
        mlp_class = TDQ_MLP

    quant_registry = {
        "snr_mean": float(snr_mean),
        "snr_std": float(snr_std),
        "method": QUANT_METHOD,
        "layers": {}
    }
    
    debug_stats = {}
    print(f"Training {QUANT_METHOD} MLPs for {len(all_linear_layers)} layers...")

    for idx, layer_name in enumerate(all_linear_layers):
        raw_stats = collector.stats[layer_name]
        if not raw_stats: continue

        try:
            reshaped_stats = np.array(raw_stats).reshape(-1, NUM_INFERENCE_STEPS)
            # Find the max activation scale across the batch for each timestep
            targets = torch.tensor(np.max(reshaped_stats, axis=0), dtype=torch.float32).to(DEVICE).view(-1, 1)
        except Exception as e:
            print(f"Skipping {layer_name}: {e}")
            continue

        mlp = mlp_class().to(DEVICE)
        initialize_mlp_bias(mlp, targets.mean())
        optimizer = optim.Adam(mlp.parameters(), lr=1e-3)

        # Train small MLP to predict the scale factor
        for _ in range(500):
            optimizer.zero_grad()
            pred = mlp(model_input)
            loss = torch.mean((torch.log(pred + 1e-9) - torch.log(targets + 1e-9)) ** 2)
            loss.backward()
            optimizer.step()

        quant_registry["layers"][layer_name] = mlp.state_dict()
        debug_stats[layer_name] = {"min": float(targets.min()), "max": float(targets.max()), "loss": loss.item()}

        if (idx + 1) % 50 == 0:
            print(f"Progress: {idx + 1}/{len(all_linear_layers)}...")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(quant_registry, SAVE_PATH)
    with open(DEBUG_JSON_PATH, "w") as f: json.dump(debug_stats, f, indent=4)
    print(f"Calibration Complete. Saved to {SAVE_PATH}")

if __name__ == "__main__":
    calibrate_and_save()