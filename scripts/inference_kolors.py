#!/usr/bin/env python3
"""
Inference script for SpaceVision Kolors LoRA

Usage:
    python scripts/inference_kolors.py --prompt "spacevision, nebula with stars"
    python scripts/inference_kolors.py --prompt "spacevision, galaxy" --output my_image.png
    python scripts/inference_kolors.py --prompt "spacevision, planet" --no-lora  # Base model only
"""

import argparse
import os
import torch
from diffusers import KolorsPipeline
from pathlib import Path

# Get the project root directory (parent of scripts/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args():
    # Default to local path, fallback to HF repo via env var
    default_lora = str(PROJECT_ROOT / "output" / "kolors-lora" / "pytorch_lora_weights.safetensors")
    if not Path(default_lora).exists():
        # Fallback to HF repo if local doesn't exist
        hf_username = os.environ.get("HF_USERNAME", "")
        if hf_username:
            default_lora = f"{hf_username}/spacevision-kolors-lora"
        else:
            default_lora = None  # Will require --lora-path or --no-lora

    parser = argparse.ArgumentParser(description="Generate images with Kolors + SpaceVision LoRA")
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
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)"
    )
    parser.add_argument(
        "--guidance", "-g",
        type=float,
        default=3.4,
        help="Guidance scale (default: 3.4)"
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

    print("Loading Kolors pipeline...")
    pipe = KolorsPipeline.from_pretrained(
        "Kwai-Kolors/Kolors-diffusers",
        torch_dtype=torch.float16,
        variant="fp16"
    )

    if not args.no_lora:
        print(f"Loading LoRA weights from {args.lora_path}...")
        pipe.load_lora_weights(args.lora_path)

    pipe = pipe.to("cuda")

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        print(f"Using seed: {args.seed}")

    print(f"Generating image...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Steps: {args.steps}, Guidance: {args.guidance}")
    print(f"  Size: {args.width}x{args.height}")

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
