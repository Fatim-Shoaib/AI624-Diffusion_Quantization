import torch
import torch.nn as nn
import os
import gc
import sys
import re
from transformers import T5Tokenizer, T5EncoderModel
from diffusers import PixArtAlphaPipeline, Transformer2DModel
from gptq_sequential_algo import GPTQSequentialAlgo

# --- Configuration ---
MODEL_ID = "PixArt-alpha/PixArt-XL-2-1024-MS" 
DEVICE = "cuda"
DTYPE = torch.float16
QUANT_BITS = 4
GROUP_SIZE = 128
CHECKPOINT_DIR = "./gptq_4block_checkpoints"

# STOP AFTER 4 BLOCKS
BLOCK_LIMIT = 4 

CALIB_PROMPTS = [
    "A large passenger airplane flying through the sky.", "A cat sitting on top of a wooden table.", "A woman playing tennis on a court.",
    "A group of people standing around a large clock.", "A baseball player swinging a bat at a ball.", "A train traveling down tracks next to a forest.",
    "A herd of elephants walking across a dry grass field.", "A man riding a skateboard down a city street.", "A plate of food with meat and vegetables.",
    "A living room with a couch and a television.", "A surfer riding a large wave in the ocean.", "A vase filled with colorful flowers on a windowsill.",
    "A zebra grazing in a field of tall grass.", "A motorcycle parked in front of a brick building.", "A close up of a pizza with pepperoni and cheese.",
    "A view of a city skyline at night with lights.", "A bathroom with a sink, toilet, and bathtub.", "A dog catching a frisbee in a park.",
    "A laptop computer sitting on a desk.", "A fire hydrant sitting on the side of a road.",
    "A small bird perched on a tree branch.", "A red double decker bus driving down a street.", "A giraffe standing next to a tall tree.",
    "A person skiing down a snowy mountain slope.", "A bowl of soup with a spoon resting in it.", "A beach with white sand and clear blue water.",
    "A traffic light hanging over a city street.", "A horse grazing in a green pasture.", "A person flying a kite in a park.",
    "A boat floating on a calm lake.", "A teddy bear sitting on a bed.", "A clock on a wall showing the time.", "A person holding a cell phone.",
    "A stop sign on a street corner.", "A person riding a bicycle.", "A cake with candles on top.", "A refrigerator filled with food.",
    "A person working on a laptop.", "A bookshelf filled with books.", "A car parked in a driveway.", "A person brushing their teeth.",
    "A dog sleeping on a rug.", "A cat looking out a window.", "A person playing a guitar.", "A group of birds flying in formation.",
    "A person eating a sandwich.", "A table set for dinner.", "A person reading a newspaper.", "A potted plant sitting on a desk.",
    "A person walking a dog on a leash."
]
CALIB_STEPS = 10 

class QuantizerParams:
    def __init__(self): self.maxq = 0; self.scale = 0; self.zero = 0
    def configure(self, bits): self.maxq = torch.tensor(2 ** bits - 1)
    def find_params(self, x, weight=False):
        dev = x.device; self.maxq = self.maxq.to(dev)
        x = x.flatten(1)
        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        tmp = (xmin == 0) & (xmax == 0); xmin[tmp] = -1; xmax[tmp] = +1
        self.scale = (xmax - xmin) / self.maxq
        self.zero = torch.round(-xmin / self.scale)
        self.scale = self.scale.unsqueeze(1); self.zero = self.zero.unsqueeze(1)
    def ready(self): return torch.is_tensor(self.scale)

class InputCapturer:
    def __init__(self): self.inputs = []
    def add_batch(self, hs, ehs, emb, ack):
        self.inputs.append({
            "hidden_states": hs.detach().cpu(),
            "encoder_hidden_states": ehs.detach().cpu() if ehs is not None else None,
            "timestep_emb": emb.detach().cpu() if emb is not None else None,
            "added_cond_kwargs": {k: v.detach().cpu() for k, v in ack.items()} if ack else {}
        })

def get_layers_in_block(block):
    layers = []
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear) and "norm" not in name:
            layers.append((name, module))
    return layers

def find_latest_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR): return None, -1
    files = os.listdir(CHECKPOINT_DIR)
    max_step = -1
    latest_file = None
    pattern = re.compile(r"model_block_(\d+)\.pt")
    for f in files:
        match = pattern.search(f)
        if match:
            step = int(match.group(1))
            if os.path.exists(os.path.join(CHECKPOINT_DIR, f"inputs_block_{step}.pt")):
                if step > max_step:
                    max_step = step
                    latest_file = os.path.join(CHECKPOINT_DIR, f)
    return latest_file, max_step

def main():
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

    print(f"Loading Model for GPTQ...")
    try:
        tokenizer = T5Tokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer", use_fast=False)
        text_encoder = T5EncoderModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=DTYPE).to(DEVICE)
        transformer = Transformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=DTYPE).to(DEVICE)
        pipe = PixArtAlphaPipeline.from_pretrained(MODEL_ID, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer, torch_dtype=DTYPE).to(DEVICE)
        pipe.set_progress_bar_config(disable=False)
    except Exception as e: print(e); return

    model = pipe.transformer
    model.eval()

    # --- RESUME LOGIC ---
    latest_ckpt, resume_block = find_latest_checkpoint()
    cached_inputs = None
    start_block = 0

    if latest_ckpt:
        print(f"\n[RESUME] Found checkpoint: {latest_ckpt}")
        state_dict = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(state_dict)
        inputs_path = os.path.join(CHECKPOINT_DIR, f"inputs_block_{resume_block}.pt")
        print(f"[RESUME] Loading cached inputs...")
        cached_inputs = torch.load(inputs_path, map_location="cpu")
        start_block = resume_block + 1
    else:
        print("\n[INIT] Starting fresh from Phase 1.")

    # --- PHASE 1: CAPTURE INITIAL INPUTS ---
    if cached_inputs is None:
        print("\n--- Phase 1: Capturing Inputs to Block 0 ---")
        capturer = InputCapturer()
        
        def catch_inputs_hook(module, args, kwargs):
            hs = args[0]
            ehs = kwargs.get('encoder_hidden_states'); ehs = args[1] if ehs is None and len(args)>1 else ehs
            emb = kwargs.get('timestep'); emb = kwargs.get('timestep_emb') if emb is None else emb; emb = args[2] if emb is None and len(args)>2 else emb
            ack = kwargs.get('added_cond_kwargs'); ack = args[3] if ack is None and len(args)>3 else ack
            capturer.add_batch(hs, ehs, emb, ack)
            return None 

        hook = model.transformer_blocks[0].register_forward_pre_hook(catch_inputs_hook, with_kwargs=True)
        
        with torch.no_grad():
            for i, prompt in enumerate(CALIB_PROMPTS):
                print(f"Prompt {i+1}/{len(CALIB_PROMPTS)}", end="\r")
                pipe(prompt, num_inference_steps=CALIB_STEPS)
        
        hook.remove()
        cached_inputs = capturer.inputs
        
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_block_-1.pt"))
        torch.save(cached_inputs, os.path.join(CHECKPOINT_DIR, "inputs_block_-1.pt"))

    del pipe.text_encoder, pipe.vae, pipe
    gc.collect(); torch.cuda.empty_cache()

    # --- PHASE 2: SEQUENTIAL GPTQ ---
    print(f"\n--- Phase 2: Sequential GPTQ (Blocks {start_block} to {BLOCK_LIMIT-1}) ---")
    
    for i in range(start_block, BLOCK_LIMIT):
        block = model.transformer_blocks[i]
        print(f"\n[Block {i}] Processing...")
        
        block.to(DEVICE)
        layers = get_layers_in_block(block)
        
        # USE GPTQ HANDLER
        handlers = {name: GPTQSequentialAlgo(layer) for name, layer in layers}
        for h in handlers.values():
            h.dev = torch.device(DEVICE); h.H = h.H.to(DEVICE) # GPU

        def get_internal_hook(h_name):
            def hook(module, inp, out):
                handlers[h_name].add_batch(inp[0]) # Capture on GPU
            return hook
        internal_hooks = [layer.register_forward_hook(get_internal_hook(name)) for name, layer in layers]
        
        # 1. Calibrate
        print(f"  Calibrating ({len(cached_inputs)} samples)...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(cached_inputs):
                hs = batch['hidden_states'].to(DEVICE)
                ehs = batch['encoder_hidden_states'].to(DEVICE) if batch['encoder_hidden_states'] is not None else None
                emb = batch['timestep_emb'].to(DEVICE) if batch['timestep_emb'] is not None else None
                ack = {k: v.to(DEVICE) for k,v in batch['added_cond_kwargs'].items()}
                block(hidden_states=hs, encoder_hidden_states=ehs, timestep=emb, added_cond_kwargs=ack)
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"    Sample {batch_idx + 1}/{len(cached_inputs)}", end="\r")
        print("")
        for h in internal_hooks: h.remove()
        
        # 2. Quantize
        print(f"  Quantizing {len(handlers)} layers (Standard GPTQ)...")
        for layer_idx, (name, layer) in enumerate(layers):
            handler = handlers[name]
            qparams = QuantizerParams()
            qparams.configure(bits=QUANT_BITS)
            
            # --- ADAPTIVE DAMPENING (Same as Qronos for Fairness) ---
            if "ff.net" in name: damp = 0.1
            else: damp = 0.01
            
            if (layer_idx + 1) % 5 == 0:
                print(f"    Layer {layer_idx + 1}/{len(layers)}", end="\r")
            
            handler.fasterquant(qparams, blocksize=128, groupsize=GROUP_SIZE, percdamp=damp)
            handler.H = None
        print("")
            
        del handlers
        gc.collect(); torch.cuda.empty_cache()
        
        # 3. Update Inputs
        print(f"  Updating inputs for next block...")
        new_cached_inputs = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(cached_inputs):
                hs = batch['hidden_states'].to(DEVICE)
                ehs = batch['encoder_hidden_states'].to(DEVICE) if batch['encoder_hidden_states'] is not None else None
                emb = batch['timestep_emb'].to(DEVICE) if batch['timestep_emb'] is not None else None
                ack = {k: v.to(DEVICE) for k,v in batch['added_cond_kwargs'].items()}
                
                output = block(hidden_states=hs, encoder_hidden_states=ehs, timestep=emb, added_cond_kwargs=ack)
                
                batch['hidden_states'] = output.detach().cpu()
                new_cached_inputs.append(batch)
                
                if (batch_idx + 1) % 100 == 0:
                    print(f"    Updating {batch_idx + 1}", end="\r")
        print("")
        
        cached_inputs = new_cached_inputs
        block.cpu() 
        
        # 4. Checkpoint
        print(f"  [CHECKPOINT] Saving after Block {i}...")
        try:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"model_block_{i}.pt"))
            torch.save(cached_inputs, os.path.join(CHECKPOINT_DIR, f"inputs_block_{i}.pt"))
            # Cleanup
            old_in = os.path.join(CHECKPOINT_DIR, f"inputs_block_{i-1}.pt")
            old_mod = os.path.join(CHECKPOINT_DIR, f"model_block_{i-1}.pt")
            if os.path.exists(old_in): os.remove(old_in)
            if os.path.exists(old_mod): os.remove(old_mod)
        except Exception as e:
            print(f"  [WARN] Save failed: {e}")

    final_path = os.path.join(CHECKPOINT_DIR, "pixart_4block_gptq.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n[DONE] 4-Block GPTQ Model saved to: {final_path}")

if __name__ == "__main__":
    main()