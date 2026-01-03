"""Generate side-by-side comparison images: base Kolors vs spacevision LoRA."""

import argparse
import os

import torch
from diffusers import KolorsPipeline
from PIL import Image, ImageDraw, ImageFont


PROMPTS = [
    "spacevision, Photograph of the International Space Station floating in space with large solar panels extended, Earth's curvature visible below showing blue ocean and white cloud formations against the blackness of space",
    "spacevision, High-resolution color photograph of a supernova remnant taken by the Hubble Space Telescope, filled with vibrant multicolored emissions of green, blue, red and yellow against a dark star-filled background with complex filament-like structures",
    "spacevision, High-resolution photograph of a satellite in space against a pitch-black background, featuring a cylindrical main body with solar panels extending horizontally showing a golden-brown hue with grid-like pattern of solar cells",
    "spacevision, NASA photograph of an astronaut performing a spacewalk outside the International Space Station, white spacesuit illuminated by sunlight against the deep black of space with Earth's blue atmosphere glowing on the horizon",
    "spacevision, Stunning photograph of Earth from low orbit showing a massive hurricane swirling over the Atlantic Ocean, spiral cloud bands clearly visible with the eye of the storm at center, thin blue atmospheric layer on the horizon",
    "spacevision, Hubble Space Telescope deep field image showing thousands of distant galaxies of various shapes and colors, spiral and elliptical galaxies scattered across a dark background revealing the vastness of the observable universe",
    "spacevision, High-resolution photograph of the Pillars of Creation in the Eagle Nebula captured by the James Webb Space Telescope, towering columns of interstellar gas and dust glowing with infrared light in shades of orange, brown and blue",
    "spacevision, NASA photograph of a SpaceX Falcon 9 rocket launching at night from Cape Canaveral, brilliant orange flame and exhaust plume illuminating the launch pad and surrounding water with streaks of light against a dark sky",
    "spacevision, Detailed photograph of the surface of Mars taken by the Curiosity rover showing reddish-brown rocky terrain with layered sedimentary formations, distant mountains on the horizon under a dusty pinkish-tan sky",
    "spacevision, Photograph of the Andromeda Galaxy taken through a large telescope, showing the spiral galaxy's bright central core surrounded by sweeping arms of stars and dust lanes with companion galaxies visible nearby against a star-filled background",
]

MODEL_ID = "Kwai-Kolors/Kolors-diffusers"
LORA_PATH = "output/kolors-lora/pytorch_lora_weights.safetensors"
OUTPUT_DIR = "output/comparisons"
SEED = 42
GUIDANCE = 3.4
STEPS = 50
SIZE = 1024


def generate(pipe, prompt, seed):
    return pipe(
        prompt=prompt,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        height=SIZE,
        width=SIZE,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]


def stitch(base_img, lora_img, prompt):
    label_h = 40
    gap = 4
    w = base_img.width * 2 + gap
    h = base_img.height + label_h
    canvas = Image.new("RGB", (w, h), (30, 30, 30))

    canvas.paste(base_img, (0, label_h))
    canvas.paste(lora_img, (base_img.width + gap, label_h))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    draw.text((base_img.width // 2, 10), "Base Kolors", fill="white", font=font, anchor="mt")
    draw.text((base_img.width + gap + base_img.width // 2, 10), "LoRA (spacevision)", fill="white", font=font, anchor="mt")

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", default=LORA_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading base Kolors pipeline...")
    pipe = KolorsPipeline.from_pretrained(MODEL_ID, variant="fp16", torch_dtype=torch.float16)
    pipe.to("cuda")

    base_images = []
    for i, prompt in enumerate(PROMPTS):
        print(f"[Base] Generating {i+1}/{len(PROMPTS)}: {prompt[:60]}...")
        img = generate(pipe, prompt, SEED + i)
        img.save(os.path.join(args.output_dir, f"base_{i}.png"))
        base_images.append(img)

    print(f"Loading LoRA weights from {args.lora_path}...")
    pipe.load_lora_weights(args.lora_path)

    lora_images = []
    for i, prompt in enumerate(PROMPTS):
        print(f"[LoRA] Generating {i+1}/{len(PROMPTS)}: {prompt[:60]}...")
        img = generate(pipe, prompt, SEED + i)
        img.save(os.path.join(args.output_dir, f"lora_{i}.png"))
        lora_images.append(img)

    for i, prompt in enumerate(PROMPTS):
        comparison = stitch(base_images[i], lora_images[i], prompt)
        path = os.path.join(args.output_dir, f"comparison_{i}.png")
        comparison.save(path)
        print(f"Saved comparison: {path}")

    print("Done!")


if __name__ == "__main__":
    main()
