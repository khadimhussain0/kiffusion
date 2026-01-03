#!/usr/bin/env python3
"""
Export trained models to HuggingFace Hub.
Handles LoRA weights, metadata, and model cards.
"""

import click
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.prompt import Confirm

console = Console()


def create_model_card(
    model_name: str,
    base_model: str,
    trigger_word: str = None,
    description: str = None,
    tags: list[str] = None,
    training_info: dict = None,
) -> str:
    """Generate a model card (README.md) for HuggingFace."""
    tags = tags or ["lora", "diffusers", "stable-diffusion"]

    frontmatter = {
        "language": ["en"],
        "tags": tags,
        "license": "apache-2.0",
        "base_model": base_model,
        "library_name": "diffusers",
    }

    yaml_str = yaml.dump(frontmatter, default_flow_style=False)

    trigger_section = ""
    if trigger_word:
        trigger_section = f"""
## Trigger Word

Use `{trigger_word}` in your prompts to activate the style/concept.

**Example prompts:**
- `a portrait photo, {trigger_word} style`
- `landscape photography, {trigger_word}`
"""

    training_section = ""
    if training_info:
        training_section = f"""
## Training Details

- **Steps:** {training_info.get('steps', 'N/A')}
- **Learning Rate:** {training_info.get('learning_rate', 'N/A')}
- **LoRA Rank:** {training_info.get('rank', 'N/A')}
- **Resolution:** {training_info.get('resolution', 'N/A')}
"""

    card = f"""---
{yaml_str}---

# {model_name}

{description or 'A LoRA model trained with Kiffusion.'}

## Model Details

- **Base Model:** [{base_model}](https://huggingface.co/{base_model})
- **Type:** LoRA
- **Format:** safetensors
{trigger_section}
## Usage

### With Diffusers

```python
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained(
    "{base_model}",
    torch_dtype=torch.float16,
)
pipe.load_lora_weights("YOUR_USERNAME/{model_name}")
pipe.to("cuda")

image = pipe("your prompt here").images[0]
```

### With ComfyUI

1. Download the `.safetensors` file
2. Place in `ComfyUI/models/loras/`
3. Use LoRA Loader node
{training_section}
## License

Apache 2.0

---

*Trained with [Kiffusion](https://github.com/khadimhussain0/kiffusion)*
"""
    return card


def convert_to_safetensors(checkpoint_path: str, output_path: str):
    """Ensure checkpoint is in safetensors format."""
    import torch
    from safetensors.torch import save_file

    cp_path = Path(checkpoint_path)

    if cp_path.suffix == ".safetensors":
        if str(cp_path) != output_path:
            shutil.copy(cp_path, output_path)
        return

    console.print(f"Converting {cp_path.suffix} to safetensors...")

    state_dict = torch.load(cp_path, map_location="cpu")
    save_file(state_dict, output_path)

    console.print("[green]Conversion complete[/green]")


def add_lora_metadata(safetensors_path: str, metadata: dict):
    """Add metadata to safetensors file."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(safetensors_path, framework="pt") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}

    save_file(tensors, safetensors_path, metadata=metadata)


@click.command()
@click.option("--checkpoint", "-c", required=True, type=click.Path(exists=True),
              help="Path to trained checkpoint")
@click.option("--output", "-o", default="./outputs/export",
              help="Export output directory")
@click.option("--name", "-n", required=True,
              help="Model name for HuggingFace")
@click.option("--base-model", "-b", required=True,
              help="Base model name (e.g., stabilityai/stable-diffusion-xl-base-1.0)")
@click.option("--trigger", "-t", default=None,
              help="Trigger word for the model")
@click.option("--description", "-d", default=None,
              help="Model description")
@click.option("--push", is_flag=True,
              help="Push to HuggingFace Hub after export")
@click.option("--repo-id", default=None,
              help="HuggingFace repo ID (username/model-name)")
@click.option("--private", is_flag=True,
              help="Create private repository")
def export(checkpoint: str, output: str, name: str, base_model: str,
           trigger: str, description: str, push: bool, repo_id: str, private: bool):
    """
    Export trained LoRA to HuggingFace-compatible format.

    Examples:
        # Export locally
        python export_hf.py -c ./outputs/checkpoint-1000 -n my-lora \\
            -b stabilityai/stable-diffusion-xl-base-1.0

        # Export and push to Hub
        python export_hf.py -c ./outputs/checkpoint-1000 -n my-lora \\
            -b stabilityai/stable-diffusion-xl-base-1.0 \\
            --push --repo-id username/my-lora
    """
    console.print("[bold]Kiffusion Export[/bold]\n")

    checkpoint_path = Path(checkpoint)
    output_path = Path(output) / name
    output_path.mkdir(parents=True, exist_ok=True)

    console.print(f"Checkpoint: {checkpoint}")
    console.print(f"Output: {output_path}")
    console.print(f"Model name: {name}")
    console.print(f"Base model: {base_model}")

    lora_file = None
    for pattern in ["*.safetensors", "pytorch_lora_weights.*", "adapter_model.*"]:
        files = list(checkpoint_path.glob(pattern))
        if files:
            lora_file = files[0]
            break

    if checkpoint_path.is_file():
        lora_file = checkpoint_path

    if not lora_file:
        console.print("[red]No LoRA weights found in checkpoint directory[/red]")
        return

    console.print(f"Found weights: {lora_file}")

    output_weights = output_path / f"{name}.safetensors"
    convert_to_safetensors(str(lora_file), str(output_weights))

    metadata = {
        "base_model": base_model,
        "trigger_word": trigger or "",
        "exported_at": datetime.now().isoformat(),
        "exported_by": "kiffusion",
    }

    config_file = checkpoint_path / "config.yaml"
    training_info = {}
    if config_file.exists():
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)
            training_info = {
                "steps": cfg.get("training", {}).get("max_train_steps"),
                "learning_rate": cfg.get("training", {}).get("learning_rate"),
                "rank": cfg.get("lora", {}).get("rank"),
                "resolution": cfg.get("dataset", {}).get("resolution"),
            }
            metadata.update({f"training_{k}": str(v) for k, v in training_info.items() if v})

    add_lora_metadata(str(output_weights), metadata)
    console.print("[green]Metadata added to weights file[/green]")

    model_card = create_model_card(
        model_name=name,
        base_model=base_model,
        trigger_word=trigger,
        description=description,
        training_info=training_info,
    )

    readme_path = output_path / "README.md"
    readme_path.write_text(model_card)
    console.print(f"[green]Model card created: {readme_path}[/green]")

    console.print("\n[bold green]Export complete![/bold green]")
    console.print(f"\nFiles in {output_path}:")
    for f in output_path.iterdir():
        size = f.stat().st_size / (1024 * 1024)
        console.print(f"  - {f.name} ({size:.2f} MB)")

    if push:
        if not repo_id:
            console.print("[red]--repo-id required for pushing to Hub[/red]")
            return

        console.print(f"\n[bold]Pushing to HuggingFace: {repo_id}[/bold]")

        if not Confirm.ask("Continue with upload?"):
            return

        try:
            from huggingface_hub import HfApi, create_repo

            api = HfApi()

            try:
                create_repo(repo_id, private=private, exist_ok=True)
            except Exception as e:
                console.print(f"[yellow]Repo creation note: {e}[/yellow]")

            api.upload_folder(
                folder_path=str(output_path),
                repo_id=repo_id,
                repo_type="model",
            )

            console.print(f"[bold green]Uploaded to https://huggingface.co/{repo_id}[/bold green]")

        except ImportError:
            console.print("[red]huggingface_hub not installed. Run: pip install huggingface_hub[/red]")
        except Exception as e:
            console.print(f"[red]Upload failed: {e}[/red]")


if __name__ == "__main__":
    export()
