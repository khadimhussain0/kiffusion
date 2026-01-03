#!/usr/bin/env python3
"""
Auto-Captioning Script for Kiffusion.
Uses JoyCaption (recommended), Florence-2, or BLIP-2 for automatic image captioning.
"""

import json
import click
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm


JOYCAPTION_PROMPTS = {
    "descriptive": "Write a descriptive caption for this image in a formal tone.",
    "descriptive_casual": "Write a descriptive caption for this image in a casual tone.",
    "straightforward": "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language.",
    "detailed": "Write a long, detailed caption for this image. Describe the subject, style, colors, composition, and mood in depth.",
    "artistic": "Describe this image as if you were an art critic, focusing on style, technique, composition, and artistic elements.",
    "training": "Write a stable diffusion prompt for this image. Focus on the subject, style, medium, colors, lighting, and composition.",
}


def load_joycaption_model(quantize: bool = False):
    """Load JoyCaption Beta One model."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model_name = "fancyfeast/llama-joycaption-beta-one-hf-llava"

    click.echo(f"  Loading {model_name}...")

    processor = AutoProcessor.from_pretrained(model_name)

    if quantize:
        # 8-bit quantization for lower VRAM
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    return processor, model


def load_blip2_model():
    """Load BLIP-2 model for captioning."""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return processor, model


def load_florence_model():
    """Load Florence-2 model for detailed captioning."""
    from transformers import AutoProcessor, AutoModelForCausalLM

    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    return processor, model


def preprocess_image_for_joycaption(image: Image.Image, target_size: int = 384) -> Image.Image:
    """Pre-resize image to avoid torchvision lanczos interpolation issues."""
    width, height = image.size
    scale = target_size / min(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    image = image.resize((new_width, new_height), Image.LANCZOS)

    left = (new_width - target_size) // 2
    top = (new_height - target_size) // 2
    image = image.crop((left, top, left + target_size, top + target_size))

    return image


def caption_with_joycaption(
    image: Image.Image,
    processor,
    model,
    caption_type: str = "descriptive",
    custom_prompt: str = None,
) -> str:
    """Generate caption using JoyCaption."""
    image = preprocess_image_for_joycaption(image)

    if custom_prompt:
        prompt = custom_prompt
    else:
        prompt = JOYCAPTION_PROMPTS.get(caption_type, JOYCAPTION_PROMPTS["descriptive"])

    convo = [
        {"role": "system", "content": "You are a helpful image captioner."},
        {"role": "user", "content": prompt},
    ]

    convo_string = processor.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[convo_string],
        images=[image],
        return_tensors="pt"
    ).to(model.device)

    if hasattr(model, 'dtype'):
        inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)
    else:
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )[0]

    caption = processor.tokenizer.decode(
        generate_ids[inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    return caption.strip()


def caption_with_blip2(image: Image.Image, processor, model, prompt: str = None) -> str:
    """Generate caption using BLIP-2."""
    if prompt:
        inputs = processor(image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
    else:
        inputs = processor(image, return_tensors="pt").to(model.device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=100)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption


def caption_with_florence(image: Image.Image, processor, model, task: str = "<MORE_DETAILED_CAPTION>") -> str:
    """Generate caption using Florence-2."""
    inputs = processor(text=task, images=image, return_tensors="pt").to(model.device, torch.float16)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=256,
        num_beams=3,
    )
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if task in caption:
        caption = caption.replace(task, "").strip()

    return caption


def add_trigger_word(caption: str, trigger: str, position: str = "prefix") -> str:
    """Add trigger word to caption."""
    if not trigger:
        return caption

    if position == "prefix":
        return f"{trigger}, {caption}"
    elif position == "suffix":
        return f"{caption}, {trigger}"
    else:
        return f"{trigger} style, {caption}"


@click.command()
@click.option("--input", "-i", "input_dir", required=True, type=click.Path(exists=True),
              help="Input directory with images")
@click.option("--output", "-o", "output_dir", default=None,
              help="Output directory for captions (default: same as input)")
@click.option("--model", "-m", type=click.Choice(["joycaption", "florence", "blip2"]),
              default="joycaption", help="Captioning model (joycaption recommended)")
@click.option("--caption-type", "-c", type=click.Choice([
              "descriptive", "descriptive_casual", "straightforward",
              "detailed", "artistic", "training"]),
              default="descriptive", help="Caption style (for JoyCaption)")
@click.option("--trigger", "-t", default=None,
              help="Trigger word to add to captions")
@click.option("--trigger-position", type=click.Choice(["prefix", "suffix", "style"]),
              default="prefix", help="Where to add trigger word")
@click.option("--prompt", "-p", default=None,
              help="Custom prompt (overrides caption-type)")
@click.option("--format", "-f", "output_format", type=click.Choice(["txt", "json", "both"]),
              default="txt", help="Output format")
@click.option("--overwrite", is_flag=True,
              help="Overwrite existing captions")
@click.option("--quantize", is_flag=True,
              help="Use 8-bit quantization (saves VRAM, JoyCaption only)")
def caption(input_dir: str, output_dir: str, model: str, caption_type: str,
            trigger: str, trigger_position: str, prompt: str, output_format: str,
            overwrite: bool, quantize: bool):
    """
    Auto-caption images using AI models.

    JoyCaption (default) is recommended for diffusion model training.
    It produces high-quality, detailed captions optimized for SD/FLUX.

    Examples:
        # JoyCaption with trigger word (recommended)
        python auto_caption.py -i ./datasets/raw -m joycaption -t "spacevision"

        # JoyCaption with training-style prompts
        python auto_caption.py -i ./datasets/raw -m joycaption -c training

        # Detailed artistic descriptions
        python auto_caption.py -i ./datasets/raw -m joycaption -c detailed

        # Florence-2 (faster, less detailed)
        python auto_caption.py -i ./datasets/raw -m florence -t "spacevision"

        # 8-bit quantization (if low on VRAM)
        python auto_caption.py -i ./datasets/raw -m joycaption --quantize

    Caption Types (JoyCaption):
        descriptive      - Formal, detailed description
        descriptive_casual - Casual, friendly tone
        straightforward  - Concise, objective style
        detailed         - Long, in-depth description
        artistic         - Art critic perspective
        training         - SD/FLUX prompt style

    VRAM Requirements:
        JoyCaption: ~17GB (or ~10GB with --quantize)
        Florence-2: ~8GB
        BLIP-2: ~8GB
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path

    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
    images = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]

    if not images:
        click.echo(f"No images found in {input_dir}")
        return

    click.echo(f"Found {len(images)} images")

    if not overwrite:
        existing = set()
        for f in output_path.glob("*.txt"):
            existing.add(f.stem)
        images = [img for img in images if img.stem not in existing]
        click.echo(f"Skipping {len(existing)} already captioned, processing {len(images)}")

    if not images:
        click.echo("All images already captioned!")
        return

    click.echo(f"\nLoading {model} model...")
    if model == "joycaption":
        processor, captioning_model = load_joycaption_model(quantize=quantize)
    elif model == "florence":
        processor, captioning_model = load_florence_model()
    elif model == "blip2":
        processor, captioning_model = load_blip2_model()

    click.echo("Model loaded!")

    if model == "joycaption":
        click.echo(f"Caption type: {caption_type}")
        if prompt:
            click.echo(f"Custom prompt: {prompt[:50]}...")

    results = []
    for img_path in tqdm(images, desc="Captioning"):
        try:
            image = Image.open(img_path).convert("RGB")

            if model == "joycaption":
                cap = caption_with_joycaption(
                    image, processor, captioning_model,
                    caption_type=caption_type, custom_prompt=prompt
                )
            elif model == "florence":
                cap = caption_with_florence(image, processor, captioning_model)
            elif model == "blip2":
                cap = caption_with_blip2(image, processor, captioning_model, prompt)
            else:
                cap = ""

            cap = add_trigger_word(cap, trigger, trigger_position)

            result = {
                "file": img_path.name,
                "caption": cap,
                "success": True,
            }

        except Exception as e:
            result = {
                "file": img_path.name,
                "caption": "",
                "success": False,
                "error": str(e),
            }

        results.append(result)

        if result["success"]:
            if output_format in ["txt", "both"]:
                caption_file = output_path / f"{img_path.stem}.txt"
                caption_file.write_text(result["caption"])

    successful = sum(1 for r in results if r["success"])
    click.echo(f"\nCaptioned {successful}/{len(images)} images")

    if output_format in ["json", "both"]:
        results_file = output_path / "captions.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"Results saved to {results_file}")

    click.echo("\nSample captions:")
    for r in results[:3]:
        if r["success"]:
            click.echo(f"  {r['file']}:")
            click.echo(f"    {r['caption'][:100]}...")


if __name__ == "__main__":
    caption()
