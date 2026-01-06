# Kiffusion

A complete pipeline for training space & sci-fi themed LoRAs — from data collection to auto-captioning to training on Kolors, SDXL, or FLUX.2.

## Quick Links

| Resource | Link |
|----------|------|
| **Dataset** | [khadim-hussain/kiffusion-space-scifi](https://huggingface.co/datasets/khadim-hussain/kiffusion-space-scifi) |
| **Kolors LoRA** | [khadim-hussain/spacevision-kolors-lora](https://huggingface.co/khadim-hussain/spacevision-kolors-lora) |
| **FLUX.2-klein LoRA** | [khadim-hussain/spacevision-flux2-lora](https://huggingface.co/khadim-hussain/spacevision-flux2-lora) |

## Features

- **Multi-source data collection** — NASA API, Wallhaven, LAION-Aesthetics scrapers
- **Auto-captioning** — JoyCaption integration with trigger word support
- **Training-ready** — SimpleTuner configs for Kolors, SDXL, and FLUX.2
- **HuggingFace export** — Dataset and model publishing scripts

## Dataset

The [kiffusion-space-scifi](https://huggingface.co/datasets/khadim-hussain/kiffusion-space-scifi) dataset contains **1,964 curated image-caption pairs**:

| Source | Count | Description |
|--------|-------|-------------|
| NASA | 915 | Real space photography (public domain) |
| Wallhaven | 845 | High-quality sci-fi & space digital art |
| LAION | 204 | Curated from LAION-2B aesthetic subset |

All images are RGB, minimum 512px, captioned with JoyCaption using the trigger word `spacevision`.

```python
# Load the dataset
from datasets import load_dataset
dataset = load_dataset("khadim-hussain/kiffusion-space-scifi")
```

## Trained Models

### Kolors LoRA

The [spacevision-kolors-lora](https://huggingface.co/khadim-hussain/spacevision-kolors-lora) is a 46.6MB LoRA adapter:

```python
from diffusers import KolorsPipeline
import torch

pipe = KolorsPipeline.from_pretrained(
    "Kwai-Kolors/Kolors-diffusers",
    torch_dtype=torch.float16
)
pipe.load_lora_weights("khadim-hussain/spacevision-kolors-lora")
pipe = pipe.to("cuda")

image = pipe(
    "spacevision, nebula with swirling purple gas clouds and bright stars",
    num_inference_steps=50,
    guidance_scale=3.4
).images[0]
```

### FLUX.2-klein LoRA

A 38MB LoRA adapter trained on FLUX.2-klein-base-9B with int8 quantization:

```python
from diffusers import Flux2KleinPipeline
from optimum.quanto import quantize, freeze, qint8
import torch

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-base-9B",
    torch_dtype=torch.bfloat16
)

# Quantize transformer (recommended, matches training)
quantize(pipe.transformer, weights=qint8)
freeze(pipe.transformer)
pipe.to("cuda")

pipe.load_lora_weights("khadim-hussain/spacevision-flux2-lora")

image = pipe(
    "spacevision, the Pillars of Creation in Eagle Nebula",
    num_inference_steps=16,
    guidance_scale=4.0,
    height=1024,
    width=1024
).images[0]
```

## Setup

```bash
git clone https://github.com/khadimhussain0/kiffusion.git
cd kiffusion

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

## Usage

### Collect Data

```bash
# NASA space imagery
python scripts/scrape_nasa.py -q "nebula" -q "galaxy" -q "supernova" -l 75

# Wallhaven sci-fi art
python scripts/scrape_wallhaven.py -q "space nebula" -q "sci-fi landscape" -l 100

# LAION aesthetic subset
python scripts/download_laion.py -k "nebula" -k "spacecraft" -l 200
```

### Caption Images

```bash
python scripts/auto_caption.py \
  -i ./datasets/raw/nasa \
  -m joycaption \
  -t "spacevision"
```

### Train with SimpleTuner

```bash
# Authenticate
huggingface-cli login
wandb login

# Select config for your target model:
#   config/config_kolors.json - Kolors (2.6B, ~12GB VRAM)
#   config/config_flux2.json  - FLUX.2-klein (9B, int8 quantized, ~24GB VRAM)

# Copy your chosen config to config.json
cp config/config_flux2.json config/config.json

# Launch training
simpletuner train
```

### Run Inference

```bash
# Generate with Kolors + LoRA
python scripts/inference_kolors.py \
  --prompt "spacevision, a nebula with swirling purple gas" \
  --output nebula.png

# Generate with FLUX.2-klein + LoRA
python scripts/inference_flux2.py \
  --prompt "spacevision, the Pillars of Creation in Eagle Nebula" \
  --output pillars.png

# Compare base model vs LoRA
python scripts/compare_models.py
```

**Inference options:**
- `--no-lora` — Run base model without LoRA
- `--seed 42` — Set random seed for reproducibility
- `--steps 20` — Number of inference steps
- `--guidance 4.0` — Guidance scale
- `--width 1024 --height 1024` — Output resolution

## Project Structure

```
kiffusion/
├── config/
│   ├── config_kolors.json          # Kolors training config
│   ├── config_flux2.json           # FLUX.2-klein training config
│   ├── multidatabackend_kolors.json # Kolors dataset config
│   └── multidatabackend_flux2.json  # FLUX.2 dataset config
├── scripts/
│   ├── scrape_nasa.py              # NASA Images API scraper
│   ├── scrape_wallhaven.py         # Wallhaven API scraper
│   ├── download_laion.py           # LAION-Aesthetics downloader
│   ├── auto_caption.py             # JoyCaption pipeline
│   ├── convert_dataset.py          # Format conversion
│   ├── inference_kolors.py         # Kolors inference script
│   ├── inference_flux2.py          # FLUX.2-klein inference script
│   ├── compare_models.py           # Base vs LoRA comparison
│   └── export_hf.py                # Dataset publishing
├── output/                         # Checkpoints and trained LoRAs
└── cache/                          # VAE and text embedding caches
```

## Supported Models

| Model | Parameters | VRAM | Precision | License |
|-------|------------|------|-----------|---------|
| Kolors | 2.6B | ~12GB | fp16/bf16 | Apache 2.0 |
| SDXL | 2.6B | ~24GB | fp16/bf16 | Open RAIL++ |
| FLUX.2-klein | 9B | ~24GB | int8-quanto | Research |

## Training Configurations

### Kolors (config_kolors.json)
- LoRA rank: 16, alpha: 16
- Learning rate: 4e-5, cosine scheduler
- Batch size: 2, gradient accumulation: 4
- Resolution: 1024px

### FLUX.2-klein (config_flux2.json)
- LoRA rank: 16, alpha: 16
- Learning rate: 1e-4, cosine scheduler
- Batch size: 1, int8-quanto quantization
- Resolution: 1024px

## License

Code: Apache 2.0

Dataset: CC-BY-NC-4.0 — see [dataset card](https://huggingface.co/datasets/khadim-hussain/kiffusion-space-scifi) for details.
