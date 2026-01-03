#!/usr/bin/env python3
"""
LAION Subset Downloader for Kiffusion.
Downloads filtered subsets from LAION-Aesthetics for specific themes.

Uses HuggingFace datasets for reliable access.
Data source: https://huggingface.co/datasets/laion/laion2B-en-aesthetic
"""

import os
import json
import hashlib
import click
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def download_image_from_url(args: tuple) -> dict:
    """Download a single image from URL."""
    url, output_dir, filename_prefix, metadata = args

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=15, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "image" not in content_type:
            return None

        content_length = int(response.headers.get("content-length", 0))
        if content_length < 10000:
            return None

        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        ext = ".jpg"
        if "png" in content_type:
            ext = ".png"
        elif "webp" in content_type:
            ext = ".webp"

        filename = f"{filename_prefix}_{url_hash}{ext}"
        filepath = output_dir / filename

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return {
            "file": filename,
            "url": url,
            "caption": metadata.get("caption", ""),
            "aesthetic_score": metadata.get("aesthetic_score", 0),
        }

    except Exception:
        return None


def search_laion_hf(keywords: list, limit: int = 100, min_aesthetic: float = 6.0) -> list:
    """
    Search LAION using HuggingFace datasets with keyword filtering.
    Uses the streaming API to avoid downloading the full dataset.
    """
    results = []

    try:
        from datasets import load_dataset

        click.echo("  Loading LAION-Aesthetics from HuggingFace (streaming)...")

        hf_token = os.getenv("HF_TOKEN")
        dataset = load_dataset(
            "laion/laion2B-en-aesthetic",
            split="train",
            streaming=True,
            token=hf_token
        )

        keywords_lower = [k.lower() for k in keywords]

        count = 0
        checked = 0
        max_check = limit * 2000

        for item in tqdm(dataset, desc="  Searching", total=max_check):
            checked += 1
            if checked > max_check:
                break

            caption = (item.get("TEXT") or "").lower()

            if any(kw in caption for kw in keywords_lower):
                aesthetic = item.get("aesthetic", 0)
                if aesthetic >= min_aesthetic:
                    url = item.get("URL", "")
                    results.append({
                        "url": url,
                        "caption": item.get("TEXT", ""),
                        "aesthetic_score": aesthetic,
                    })
                    count += 1

                    if count >= limit:
                        break

        click.echo(f"  Found {len(results)} matching images")

    except ImportError:
        click.echo("  Error: 'datasets' library not installed. Run: pip install datasets")
    except Exception as e:
        click.echo(f"  Error loading dataset: {e}")

    return results


@click.command()
@click.option("--keywords", "-k", multiple=True, required=True,
              help="Keywords to search in captions (can use multiple)")
@click.option("--output", "-o", default="./datasets/raw/laion",
              help="Output directory")
@click.option("--limit", "-l", default=100, type=int,
              help="Max images to download")
@click.option("--workers", "-w", default=8, type=int,
              help="Download workers")
@click.option("--min-aesthetic", "-a", default=6.5, type=float,
              help="Minimum aesthetic score (1-10)")
def download_laion(keywords: tuple, output: str, limit: int, workers: int, min_aesthetic: float):
    """
    Download images from LAION-Aesthetics filtered by keywords.

    Examples:
        python download_laion.py -k "nebula" -k "space" -l 100
        python download_laion.py -k "sci-fi" -k "spacecraft" -l 50
        python download_laion.py -k "galaxy" -k "cosmic" -a 7.0 -l 100
    """
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"\nSearching LAION for keywords: {', '.join(keywords)}")

    results = search_laion_hf(list(keywords), limit=limit, min_aesthetic=min_aesthetic)

    if not results:
        click.echo("\nNo results found! Try different keywords or lower aesthetic threshold.")
        click.echo("\nTip: Use broad keywords like 'nebula', 'galaxy', 'sci-fi'")
        return

    click.echo(f"\nDownloading {len(results)} images...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_tasks = []
    for i, r in enumerate(results):
        prefix = f"laion_{timestamp}_{i:04d}"
        download_tasks.append((r["url"], output_dir, prefix, r))

    downloaded = []
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_image_from_url, task): task for task in download_tasks}

        with tqdm(total=len(futures), desc="Downloading") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    downloaded.append(result)
                else:
                    failed += 1
                pbar.update(1)

    click.echo(f"\nDownloaded {len(downloaded)}/{len(results)} images ({failed} failed/unavailable)")

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump({
            "source": "LAION-Aesthetics-6.5+",
            "keywords": list(keywords),
            "min_aesthetic_score": min_aesthetic,
            "downloaded_at": datetime.now().isoformat(),
            "images": downloaded,
        }, f, indent=2)

    click.echo(f"Metadata saved to {metadata_file}")

    if downloaded:
        avg_aesthetic = sum(d.get("aesthetic_score", 0) for d in downloaded) / len(downloaded)
        click.echo(f"\nStatistics:")
        click.echo(f"  Average aesthetic score: {avg_aesthetic:.2f}")

        click.echo("\nSample captions:")
        for d in downloaded[:5]:
            if d.get("caption"):
                click.echo(f"  - {d['caption'][:80]}...")

    click.echo("\nDone!")


if __name__ == "__main__":
    download_laion()
