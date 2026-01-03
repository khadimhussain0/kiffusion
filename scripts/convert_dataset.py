#!/usr/bin/env python3
"""
Convert metadata.jsonl dataset to SimpleTuner format.
Creates .txt sidecar caption files alongside each image.
"""

import json
from pathlib import Path
from rich.console import Console

console = Console()


def convert(metadata_jsonl: str, image_dir: str):
    """Read metadata.jsonl and create .txt caption files next to each image."""
    metadata_path = Path(metadata_jsonl)
    img_dir = Path(image_dir)

    if not metadata_path.exists():
        console.print(f"[red]metadata.jsonl not found: {metadata_path}[/red]")
        return

    created = 0
    skipped = 0
    missing = 0

    with open(metadata_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            file_name = entry["file_name"]
            caption = entry["text"]

            # file_name is like "train/nasa_0000.jpg" - resolve relative to image_dir parent
            img_path = metadata_path.parent / file_name
            if not img_path.exists():
                # Try directly in image_dir
                img_path = img_dir / Path(file_name).name
            if not img_path.exists():
                missing += 1
                continue

            # Create .txt sidecar with same stem
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                skipped += 1
                continue

            txt_path.write_text(caption)
            created += 1

    console.print(f"[green]Created {created} caption files[/green]")
    if skipped:
        console.print(f"[yellow]Skipped {skipped} (already exist)[/yellow]")
    if missing:
        console.print(f"[red]Missing {missing} images[/red]")


if __name__ == "__main__":
    import sys

    metadata = sys.argv[1] if len(sys.argv) > 1 else "datasets/hf_upload/metadata.jsonl"
    img_dir = sys.argv[2] if len(sys.argv) > 2 else "datasets/hf_upload/train"

    console.print(f"[bold]Converting dataset to SimpleTuner format[/bold]")
    console.print(f"  metadata: {metadata}")
    console.print(f"  images:   {img_dir}")
    convert(metadata, img_dir)
