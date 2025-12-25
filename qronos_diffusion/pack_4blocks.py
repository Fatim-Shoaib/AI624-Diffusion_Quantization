import torch
import os
import re

# --- CONFIG ---
QRONOS_INPUT = "./qronos_4block_checkpoints/pixart_4block_qronos.pt"
GPTQ_INPUT = "./gptq_4block_checkpoints/pixart_4block_gptq.pt"
QRONOS_PACKED = "./qronos_4block_checkpoints/pixart_4block_qronos_packed.pt"
GPTQ_PACKED = "./gptq_4block_checkpoints/pixart_4block_gptq_packed.pt"

QUANT_BITS = 4
GROUP_SIZE = 128 
BLOCK_LIMIT = 4 # Only pack blocks 0-3

SKIP_KEYWORDS = ['time_embed', 'label_embed', 'pos_embed', 'context_embedder', 'time_text_embed', 'norm']

def pack_weights_groupwise(param, group_size):
    w_flat = param.flatten()
    if w_flat.numel() % group_size != 0:
        w_flat = torch.cat([w_flat, torch.zeros(group_size - (w_flat.numel() % group_size), device=param.device, dtype=param.dtype)])
    w_groups = w_flat.view(-1, group_size).to(torch.float32)
    w_min = w_groups.min(dim=1, keepdim=True)[0]
    w_max = w_groups.max(dim=1, keepdim=True)[0]
    maxq = 2**QUANT_BITS - 1
    scale = (w_max - w_min) / maxq
    scale = torch.clamp(scale, min=1e-8) 
    zero = torch.round(-w_min / scale)
    val = w_groups / scale + zero
    val = torch.clamp(torch.round(val), 0, maxq)
    int_weights = val.to(torch.uint8)
    flat_ints = int_weights.flatten()
    packed = (flat_ints[0::2] << 4) | (flat_ints[1::2])
    return packed, scale, zero, param.shape

def is_target_block(name):
    match = re.search(r"transformer_blocks\.(\d+)\.", name)
    if match:
        block_idx = int(match.group(1))
        if block_idx < BLOCK_LIMIT:
            return True
    return False

def pack_file(in_path, out_path):
    if not os.path.exists(in_path): print(f"Missing {in_path}"); return
    print(f"Packing {in_path}...")
    state_dict = torch.load(in_path, map_location="cpu")
    packed_dict = {}
    packed_count = 0
    for name, param in state_dict.items():
        is_sensitive = False
        for keyword in SKIP_KEYWORDS:
            if keyword in name: is_sensitive = True; break
        
        if param.dim() > 1 and "weight" in name and not is_sensitive and is_target_block(name):
            pd, sc, ze, sh = pack_weights_groupwise(param, GROUP_SIZE)
            packed_dict[name + ".packed_weight"] = pd
            packed_dict[name + ".scales"] = sc 
            packed_dict[name + ".zeros"] = ze   
            packed_dict[name + ".shape"] = torch.tensor(sh)
            packed_count += 1
        else:
            packed_dict[name] = param
    print(f"  Packed {packed_count} layers.")
    torch.save(packed_dict, out_path)

def main():
    pack_file(QRONOS_INPUT, QRONOS_PACKED)
    pack_file(GPTQ_INPUT, GPTQ_PACKED)

if __name__ == "__main__":
    main()