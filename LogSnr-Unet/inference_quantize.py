import torch
import torch.nn as nn
import os
import gc
import numpy as np
import lpips
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from networks import SNR_TDQ_MLP, TDQ_MLP
from scheduler import NoiseScheduler
from quantize import QLayer, QuantGlobalContext

# --- Configuration ---
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "legacy/quant_checkpoint_tdq.pt" 
OUTPUT_DIR = "legacy/results_comparison"
ABITS = 4  
WBITS = 8

TOTAL_IMAGES = 64     
BATCH_SIZE = 4        
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 1.0  
SEED = 42
SCALE_MULTIPLIER = 1.0
SKIP_PATTERNS = ["time", "emb", "norm", "to_k", "to_v"]

# --- Wrappers ---

class SNRWrapper(nn.Module):
    def __init__(self, mlp, snr_mean, snr_std):
        super().__init__()
        self.mlp = mlp
        self.register_buffer('mean', torch.tensor(snr_mean))
        self.register_buffer('std', torch.tensor(snr_std + 1e-6))
    
    def forward(self, raw_snr):
        norm_snr = (raw_snr - self.mean) / self.std
        return torch.abs(self.mlp(norm_snr)) * SCALE_MULTIPLIER + 1e-9

class TDQWrapper(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, raw_timestep):
        norm_t = raw_timestep / 1000.0
        if norm_t.dtype != torch.float32:
            norm_t = norm_t.float()
        return torch.abs(self.mlp(norm_t)) * SCALE_MULTIPLIER + 1e-9

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_module_by_name(root, dotted_path):
    cur = root
    for p in dotted_path.split('.'):
        cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
    return cur

def set_nested_item(root, path, value):
    parts = path.split('.')
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    if parts[-1].isdigit(): parent[int(parts[-1])] = value
    else: setattr(parent, parts[-1], value)

def patch_model(pipeline, checkpoint_data, global_context):
    ckpt_method = checkpoint_data.get("method", "tdq")
    print(f"Loading checkpoint trained with method: {ckpt_method.upper()}")
    
    layers_data = checkpoint_data["layers"]
    snr_mean = checkpoint_data.get("snr_mean", 0.0)
    snr_std = checkpoint_data.get("snr_std", 1.0)

    patched_count = 0
    for layer_name, mlp_state_dict in layers_data.items():
        if any(skip in layer_name for skip in SKIP_PATTERNS): continue
        try:
            original_module = get_module_by_name(pipeline.unet, layer_name)
            if not isinstance(original_module, nn.Linear): continue

            if "snr" in ckpt_method:
                mlp = SNR_TDQ_MLP().to(DEVICE)
                mlp.load_state_dict(mlp_state_dict)
                inference_mlp = SNRWrapper(mlp, snr_mean, snr_std)
                method_type = "snr"
            else:
                mlp = TDQ_MLP().to(DEVICE)
                mlp.load_state_dict(mlp_state_dict)
                inference_mlp = TDQWrapper(mlp)
                method_type = "tdq"
            
            q_layer = QLayer(original_module, inference_mlp, abits=ABITS, wbits=WBITS)
            q_layer.context = global_context
            q_layer.quantize_w = True
            q_layer.quantize_act = True

            set_nested_item(pipeline.unet, layer_name, q_layer)
            patched_count += 1
        except Exception as e: 
            continue
            
    print(f"Successfully patched {patched_count} layers using {method_type} logic.")
    return pipeline, method_type

def generate_batch(pipe, batch_size, noise_sched, method_type="tdq", context=None, is_quantized=False, start_idx=0):
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + start_idx)
    latents = torch.randn((batch_size, 4, 64, 64), device=DEVICE, dtype=torch.float16, generator=generator)

    uncond_input = pipe.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    emb = torch.cat([pipe.text_encoder(uncond_input.input_ids.to(DEVICE))[0]] * 2)

    for t in pipe.scheduler.timesteps:
        if is_quantized and context is not None:
            t_scalar = t.clone().detach().to(DEVICE).view(1)
            if "snr" in method_type:
                val = noise_sched.get_log_snr(t_scalar)
            else:
                val = t_scalar 

            context.set_current_snr(val)

        latent_in = torch.cat([latents] * 2)
        latent_in = pipe.scheduler.scale_model_input(latent_in, t)
        
        noise_pred = pipe.unet(latent_in, t, encoder_hidden_states=emb).sample
        u, c = noise_pred.chunk(2)
        noise_pred = u + GUIDANCE_SCALE * (c - u)
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / pipe.vae.config.scaling_factor
    images = pipe.vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
    return (images * 255).round().astype("uint8")

@torch.no_grad()
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    noise_sched = NoiseScheduler(num_timesteps=1000, schedule="linear")

    print(f"--- Stage 1: Baseline Generation ---")
    base_pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    base_pipe.scheduler = DDIMScheduler.from_config(base_pipe.scheduler.config)
    base_pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    
    for i in range(0, TOTAL_IMAGES, BATCH_SIZE):
        if os.path.exists(os.path.join(OUTPUT_DIR, f"baseline_{i}.png")): continue
        print(f"Generating Baseline Batch {i//BATCH_SIZE + 1}...")
        imgs = generate_batch(base_pipe, min(BATCH_SIZE, TOTAL_IMAGES - i), noise_sched, is_quantized=False, start_idx=i)
        for j, img_arr in enumerate(imgs):
            Image.fromarray(img_arr).save(os.path.join(OUTPUT_DIR, f"baseline_{i+j}.png"))
    
    del base_pipe
    flush()

    print(f"\n--- Stage 2: Quantized Generation ---")
    if not os.path.exists(CHECKPOINT_PATH): 
        raise FileNotFoundError(f"Missing {CHECKPOINT_PATH}. Run calibrate.py first.")

    quant_pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    quant_pipe.scheduler = DDIMScheduler.from_config(quant_pipe.scheduler.config)
    quant_pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS)

    context = QuantGlobalContext()
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    quant_pipe, method_detected = patch_model(quant_pipe, ckpt, context)
    
    for i in range(0, TOTAL_IMAGES, BATCH_SIZE):
        print(f"Generating Quantized Batch {i//BATCH_SIZE + 1}...")
        imgs = generate_batch(quant_pipe, min(BATCH_SIZE, TOTAL_IMAGES - i), noise_sched, 
                              method_type=method_detected, context=context, is_quantized=True, start_idx=i)
        for j, img_arr in enumerate(imgs):
            Image.fromarray(img_arr).save(os.path.join(OUTPUT_DIR, f"quantized_{i+j}.png"))

    del quant_pipe, ckpt
    flush()

    print("\n--- Stage 3: Calculating LPIPS ---")
    lpips_model = lpips.LPIPS(net='alex').to(DEVICE)
    scores = []
    
    for i in range(TOTAL_IMAGES):
        p_base = os.path.join(OUTPUT_DIR, f"baseline_{i}.png")
        p_quant = os.path.join(OUTPUT_DIR, f"quantized_{i}.png")
        if not os.path.exists(p_base) or not os.path.exists(p_quant): continue
        
        t_b = (torch.from_numpy(np.array(Image.open(p_base).convert("RGB"))/255.0).permute(2,0,1).unsqueeze(0).float().to(DEVICE) * 2 - 1)
        t_q = (torch.from_numpy(np.array(Image.open(p_quant).convert("RGB"))/255.0).permute(2,0,1).unsqueeze(0).float().to(DEVICE) * 2 - 1)
        
        scores.append(lpips_model(t_b, t_q).item())

    if scores:
        print("\n" + "="*40)
        print(f"CHECKPOINT: {CHECKPOINT_PATH}")
        print(f"METHOD DETECTED: {method_detected.upper()}")
        print(f"BITS: A{ABITS} / W{WBITS}")
        print(f"AVERAGE LPIPS: {np.mean(scores):.5f}")
        print("="*40)

if __name__ == "__main__":
    run()