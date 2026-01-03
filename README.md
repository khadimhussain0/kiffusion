# Kiffusion

A complete pipeline for training space & sci-fi themed LoRAs — from data collection to auto-captioning to training on Kolors, SDXL, or FLUX.

## Quick Links

| Resource | Link |
|----------|------|
| **Dataset** | [khadim-hussain/kiffusion-space-scifi](https://huggingface.co/datasets/khadim-hussain/kiffusion-space-scifi) |
| **Trained LoRA** | [khadim-hussain/spacevision-kolors-lora](https://huggingface.co/khadim-hussain/spacevision-kolors-lora) |

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

## Trained Model

The [spacevision-kolors-lora](https://huggingface.co/khadim-hussain/spacevision-kolors-lora) is a 46.6MB LoRA adapter trained on this dataset:

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

# Edit config/config.json and config/multidatabackend.json as needed

# Launch training
simpletuner train
```

The default config trains a Kolors LoRA (rank 16, cosine scheduler, 1024px).

## Project Structure

```
kiffusion/
├── config/
│   ├── config.json               # Training hyperparameters
│   └── multidatabackend.json     # Dataset configuration
├── scripts/
│   ├── scrape_nasa.py            # NASA Images API scraper
│   ├── scrape_wallhaven.py       # Wallhaven API scraper
│   ├── download_laion.py         # LAION-Aesthetics downloader
│   ├── auto_caption.py           # JoyCaption pipeline
│   ├── convert_dataset.py        # Format conversion
│   ├── compare_models.py         # Base vs LoRA comparison
│   ├── push_lora_to_hf.py        # Model publishing
│   └── export_hf.py              # Dataset publishing
└── output/                       # Checkpoints and samples
```

## Supported Models

| Model | Parameters | VRAM | License |
|-------|------------|------|---------|
| Kolors | 2.6B | ~12GB | Apache 2.0 |
| SDXL | 2.6B | ~24GB | Open RAIL++ |
| FLUX.2 [dev] | 9B/32B | ~32GB | Non-Commercial |

Change `model_family` in `config/config.json` to switch models.

## License

Code: Apache 2.0

Dataset: CC-BY-NC-4.0 — see [dataset card](https://huggingface.co/datasets/khadim-hussain/kiffusion-space-scifi) for details.
