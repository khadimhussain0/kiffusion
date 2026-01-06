#!/usr/bin/env python3
"""
Inference script for SpaceVision FLUX.2-klein LoRA

Usage:
    python scripts/inference_flux2.py --prompt "spacevision, nebula with stars"
    python scripts/inference_flux2.py --prompt "spacevision, galaxy" --output my_image.png
    python scripts/inference_flux2.py --prompt "spacevision, planet" --no-lora  # Base model only
    python scripts/inference_flux2.py --prompt "spacevision, nebula" --no-quantize  # Full precision

Note: Requires ~24GB VRAM with int8 quantization, ~48GB without.
"""

import argparse
import torch
from diffusers import Flux2KleinPipeline
from pathlib import Path
import os

# Get the project root directory (parent of scripts/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with FLUX.2-klein + SpaceVision LoRA")
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Text prompt (include 'spacevision' trigger word for LoRA style)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.png",
        help="Output image path (default: output.png)"
    )
    default_lora = str(PROJECT_ROOT / "output" / "flux2-lora" / "pytorch_lora_weights.safetensors")
    parser.add_argument(
        "--lora-path",
        type=str,
        default=default_lora,
        help="Path to LoRA weights (local path or HuggingFace repo)"
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Run without LoRA (base model only)"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable int8 quantization (requires more VRAM)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=16,
        help="Number of inference steps (default: 16)"
    )
    parser.add_argument(
        "--guidance", "-g",
        type=float,
        default=4.0,
        help="Guidance scale (default: 4.0)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading FLUX.2-klein pipeline...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-9B",
        torch_dtype=torch.bfloat16,
    )

    if not args.no_quantize:
        print("Quantizing transformer to int8...")
        from optimum.quanto import quantize, freeze, qint8
        quantize(pipe.transformer, weights=qint8)
        freeze(pipe.transformer)

    pipe.to("cuda")

    if not args.no_lora:
        print(f"Loading LoRA weights from {args.lora_path}...")
        pipe.load_lora_weights(args.lora_path)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        print(f"Using seed: {args.seed}")

    print(f"Generating image...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Steps: {args.steps}, Guidance: {args.guidance}")
    print(f"  Size: {args.width}x{args.height}")
    print(f"  Quantized: {not args.no_quantize}")

    image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        generator=generator,
    ).images[0]

    output_path = Path(args.output)
    if output_path.parent and str(output_path.parent) != ".":
        output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
