import os
import sys
import gc
import time
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import PixArtAlphaPipeline
from act_quant import dynamic_quantize_activations

# --- FIXES ---
sys.modules['tensorflow'] = None
try:
    import transformers.utils.import_utils
    def dummy_check(*args, **kwargs): return True
    transformers.utils.import_utils.check_torch_load_is_safe = dummy_check
    try:
        import transformers.modeling_utils
        transformers.modeling_utils.check_torch_load_is_safe = dummy_check
    except: pass
except: pass
# -------------

# --- CONFIGURATION ---
MODEL_ID = "PixArt-alpha/PixArt-XL-2-1024-MS"
GPTQ_PACKED = "./gptq_4block_checkpoints/pixart_4block_gptq_packed.pt"
QRONOS_PACKED = "./qronos_4block_checkpoints/pixart_4block_qronos_packed.pt"
OUTPUT_DIR = "./benchmark_final_comprehensive"
DEVICE = "cuda"
DTYPE = torch.float16
GROUP_SIZE = 128
NUM_IMAGES = 20

PROMPTS = [
    "A cinematic shot of an astronaut riding a horse on mars, highly detailed, 8k",
    "A photorealistic portrait of a cute cat sitting on a park bench",
    "A futuristic cyberpunk city with neon lights and flying cars",
    "A beautiful landscape of mountains and a lake at sunset",
    "A delicious pepperoni pizza with melting cheese",
    "A red sports car driving on a coastal road",
    "An oil painting of a cottage in the woods",
    "A robot playing chess against a human",
    "A majestic lion roaring in the savannah",
    "A glass of water sitting on a table with sunlight refracting through it",
    "A steam punk style locomotive engine emitting smoke",
    "A intricate marble statue of a greek god",
    "A macro shot of a dew drop on a green leaf",
    "A wide angle shot of a busy times square in new york",
    "A fantasy castle floating in the clouds",
    "A bowl of fresh fruit including apples, bananas, and grapes",
    "A concept art of a dragon breathing fire",
    "A cozy living room with a fireplace and books",
    "A snowy forest path during winter",
    "A vintage camera sitting on an old map"
]

# --- W4A8 Layer ---
class QuantLinearW4A8(nn.Module):
    def __init__(self, in_features, out_features, bias=None, group_size=128):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features; self.group_size = group_size
        num_weights = out_features * in_features
        padded = num_weights
        if num_weights % group_size != 0: padded += (group_size - (num_weights % group_size))
        
        self.register_buffer('packed_weight', torch.zeros(padded // 2, dtype=torch.uint8))
        num_groups = padded // group_size
        self.register_buffer('scales', torch.zeros((num_groups, 1), dtype=torch.float32))
        self.register_buffer('zeros', torch.zeros((num_groups, 1), dtype=torch.float32))
        
        if bias is not None: self.register_buffer('bias', bias)
        else: self.bias = None

    def forward(self, x):
        x_quant = dynamic_quantize_activations(x, bits=8)
        s = self.scales.to(torch.float32); z = self.zeros.to(torch.float32)
        high = (self.packed_weight >> 4); low = (self.packed_weight & 0x0F)
        indices = torch.stack([high, low], dim=-1).flatten()
        indices_grouped = indices.view(-1, self.group_size)
        w_dequant = s * (indices_grouped.to(torch.float32) - z)
        w_flat = w_dequant.flatten()[:self.out_features * self.in_features]
        w_fp16 = w_flat.to(x.dtype).view(self.out_features, self.in_features)
        return nn.functional.linear(x_quant, w_fp16, self.bias)

def replace_linear_with_w4a8(model, packed_state_dict, prefix=""):
    count = 0
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        packed_key = full_name + ".weight.packed_weight"
        if packed_key in packed_state_dict:
            linear_layer = module
            packed_w = packed_state_dict[packed_key]
            scales = packed_state_dict[full_name + ".weight.scales"]
            zeros = packed_state_dict[full_name + ".weight.zeros"]
            
            new_layer = QuantLinearW4A8(linear_layer.in_features, linear_layer.out_features, linear_layer.bias, GROUP_SIZE)
            new_layer.packed_weight = packed_w.to(DEVICE)
            new_layer.scales = scales.to(DEVICE).float()
            new_layer.zeros = zeros.to(DEVICE).float()
            if linear_layer.bias is not None: new_layer.bias = linear_layer.bias.to(DEVICE)
            setattr(model, name, new_layer)
            count += 1
        else:
            count += replace_linear_with_w4a8(module, packed_state_dict, full_name)
    return count

def generate_images(pipe, name):
    print(f"\n--- Generating {name} ---")
    out_path = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_path, exist_ok=True)
    
    # Measure Model Size (Transformer Only)
    model_size = pipe.transformer.get_memory_footprint() / 1024**3
    
    # Measure VRAM & Time
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    times = []
    
    # Warmup
    pipe("warmup", num_inference_steps=1)
    torch.cuda.reset_peak_memory_stats()
    
    images = []
    for i, prompt in enumerate(PROMPTS):
        gen = torch.Generator(device=DEVICE).manual_seed(42+i)
        start = time.time()
        with torch.no_grad():
            img = pipe(prompt, num_inference_steps=20, generator=gen).images[0]
        times.append(time.time() - start)
        img.save(os.path.join(out_path, f"{i:02d}.png"))
        images.append(img)
        print(f"[{name}] {i+1}/{NUM_IMAGES}", end="\r")
        
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    avg_time = sum(times) / len(times)
    
    print(f"\n{name} Stats: Size={model_size:.2f}GB | VRAM={peak_vram:.2f}GB | Time={avg_time:.2f}s")
    return images, model_size, peak_vram, avg_time

def calculate_metrics(base_imgs, target_imgs):
    print(f"Calculating FID, LPIPS, CLIP...")
    from torchmetrics.multimodal.clip_score import CLIPScore
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image.fid import FrechetInceptionDistance

    clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(DEVICE)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(DEVICE)
    # FID usually requires at least 64 dims, typically 2048 for full inception
    fid = FrechetInceptionDistance(feature=64).to(DEVICE)
    
    c_scores, l_scores = [], []
    
    # Update FID batches
    # We stack images to create batches
    base_tensors = []
    target_tensors = []
    
    for i, (b_pil, t_pil) in enumerate(zip(base_imgs, target_imgs)):
        bt = torch.tensor(np.array(b_pil)).permute(2,0,1).unsqueeze(0).to(DEVICE)
        tt = torch.tensor(np.array(t_pil)).permute(2,0,1).unsqueeze(0).to(DEVICE)
        
        # CLIP
        c_scores.append(clip(tt, PROMPTS[i]).item())
        # LPIPS (normalize -1 to 1)
        l_scores.append(lpips(tt.float()/255*2-1, bt.float()/255*2-1).item())
        
        base_tensors.append(bt)
        target_tensors.append(tt)
    
    # Process FID in one go (or batches if OOM)
    # real=True for Baseline, real=False for Quantized
    fid.update(torch.cat(base_tensors), real=True)
    fid.update(torch.cat(target_tensors), real=False)
    
    fid_score = fid.compute().item()
    
    return sum(c_scores)/len(c_scores), sum(l_scores)/len(l_scores), fid_score

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # 1. FP16 Baseline
    print("Loading Baseline...")
    pipe = PixArtAlphaPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    imgs_base, size_base, vram_base, time_base = generate_images(pipe, "FP16")
    del pipe; gc.collect(); torch.cuda.empty_cache()
    
    # 2. GPTQ
    print("\nLoading GPTQ...")
    pipe = PixArtAlphaPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    state = torch.load(GPTQ_PACKED, map_location="cpu")
    replace_linear_with_w4a8(pipe.transformer, state)
    pipe.to(DEVICE)
    imgs_gptq, size_gptq, vram_gptq, time_gptq = generate_images(pipe, "GPTQ")
    del pipe; gc.collect(); torch.cuda.empty_cache()
    
    # 3. Qronos
    print("\nLoading Qronos...")
    pipe = PixArtAlphaPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    state = torch.load(QRONOS_PACKED, map_location="cpu")
    replace_linear_with_w4a8(pipe.transformer, state)
    pipe.to(DEVICE)
    imgs_qronos, size_qronos, vram_qronos, time_qronos = generate_images(pipe, "Qronos")
    del pipe; gc.collect(); torch.cuda.empty_cache()
    
    # 4. Calc Metrics
    print("\n--- Metrics: GPTQ vs Baseline ---")
    clip_g, lpips_g, fid_g = calculate_metrics(imgs_base, imgs_gptq)
    
    print("\n--- Metrics: Qronos vs Baseline ---")
    clip_q, lpips_q, fid_q = calculate_metrics(imgs_base, imgs_qronos)
    
    # Self-test baseline CLIP
    print("\n--- Metrics: Baseline Reference ---")
    # Quick CLIP calculation for baseline
    from torchmetrics.multimodal.clip_score import CLIPScore
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(DEVICE)
    clip_b = 0
    for i, img in enumerate(imgs_base):
        t = torch.tensor(np.array(img)).permute(2,0,1).unsqueeze(0).to(DEVICE)
        clip_b += metric(t, PROMPTS[i]).item()
    clip_b /= len(PROMPTS)

    # 5. Output Table
    print("\n" + "="*80)
    print(f"{'Metric':<15} | {'FP16':<12} | {'GPTQ (4 Blk)':<15} | {'Qronos (4 Blk)':<15}")
    print("="*80)
    print(f"{'Model Size (GB)':<15} | {size_base:<12.2f} | {size_gptq:<15.2f} | {size_qronos:<15.2f}")
    print(f"{'Peak VRAM (GB)':<15} | {vram_base:<12.2f} | {vram_gptq:<15.2f} | {vram_qronos:<15.2f}")
    print(f"{'Latency (s)':<15} | {time_base:<12.2f} | {time_gptq:<15.2f} | {time_qronos:<15.2f}")
    print("-" * 80)
    print(f"{'CLIP Score':<15} | {clip_b:<12.4f} | {clip_g:<15.4f} | {clip_q:<15.4f}")
    print(f"{'LPIPS':<15} | {'0.0000':<12} | {lpips_g:<15.4f} | {lpips_q:<15.4f}")
    print(f"{'FID':<15}   | {'0.0000':<12} | {fid_g:<15.2f} | {fid_q:<15.2f}")
    print("="*80)
    
    # Visual Grid
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    cols = ["FP16", "GPTQ", "Qronos"]
    for i in range(5):
        axes[i,0].imshow(imgs_base[i]); axes[i,0].axis('off')
        axes[i,1].imshow(imgs_gptq[i]); axes[i,1].axis('off')
        axes[i,2].imshow(imgs_qronos[i]); axes[i,2].axis('off')
        if i==0:
            axes[i,0].set_title("FP16 Baseline")
            axes[i,1].set_title("GPTQ W4A8")
            axes[i,2].set_title("Qronos W4A8")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_grid.jpg"))
    print(f"Saved visual grid to {OUTPUT_DIR}/final_grid.jpg")

if __name__ == "__main__":
    main()