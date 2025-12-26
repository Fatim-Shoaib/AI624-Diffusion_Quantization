# Advanced Quantization Methods for Diffusion Models

This repository contains the implementation of post-training quantization (PTQ) techniques for Diffusion Transformers (DiTs) and UNet-based diffusion models. It focuses on maintaining high fidelity at low-bit precisions (W4A4/W4A8) through Log-SNR calibration and second-order optimization.

## Core Components

### 1. Log-SNR Calibration (`LogSnr-DiT` & `LogSnr-Unet`)
Introduces **Log-SNR Dependent Quantization (Log-SNR TDQ)**. Instead of traditional timestep-based conditioning, this method optimizes activation quantization scales based on the log-signal-to-noise ratio, providing better adaptation to the noise schedule.
*   **Target Models:** PixArt-Alpha (DiT), Stable Diffusion v1.5 (UNet).
*   **Key Scripts:** `calibrate_snr.py`, `evaluate_compare.py`.

### 2. Qronos Algorithm (`qronos_diffusion`)
A high-performance quantization framework for 4-bit weights and 4/8-bit activations. Qronos utilizes a global correction step and error diffusion to mitigate quantization noise in deep transformer blocks.
*   **Methods:** Qronos W4A4, GPTQ-Sequential.
*   **Precision:** W4A8 (Weights: 4-bit, Activations: 8-bit).
*   **Benchmark:** Comprehensive evaluation against FP16 baselines using FID, CLIP, and LPIPS.

## Repository Structure

```text
├── LogSnr-DiT/          # Log-SNR calibration for Diffusion Transformers (PixArt)
├── LogSnr-Unet/         # Log-SNR and TDQ calibration for SD v1.5
└── qronos_diffusion/    # Qronos & GPTQ algorithms for low-bit diffusion
```

## Quick Start

### Installation
```bash
pip install torch torchvision transformers diffusers accelerate torchmetrics lpips
```

### Quantization & Calibration
To run Log-SNR calibration on PixArt-Alpha:
```bash
python LogSnr-DiT/calibrate_snr.py --abits 8 --wbits 8 --save_path logsnr_checkpoints.pt
```

To run Qronos 4-block quantization:
```bash
python qronos_diffusion/quantize_qronos_4blocks.py
```

### Evaluation
Compare quantized models against FP16 baselines:
```bash
python qronos_diffusion/benchmark_final.py
```

## Comparative Metrics
The included scripts evaluate performance based on:
*   **LPIPS:** Perceptual similarity to the floating-point baseline.
*   **CLIP Score:** Semantic alignment between generated images and text prompts.
*   **FID:** Distributional quality (requires 2048-dim features for full accuracy).
*   **Hardware Efficiency:** Latency (s), Peak VRAM (GB), and Model Size (GB).

## Implementation Details
*   **TDQ MLP:** Uses Fourier embeddings to map temporal/SNR conditions to quantization scales.
*   **Sequential Block Quantization:** Processes transformer blocks one by one, caching inputs to minimize memory overhead during calibration.
*   **Adaptive Dampening:** Implements per-layer Hessian dampening to stabilize second-order optimization in sensitive layers (e.g., Feed-Forward networks).
