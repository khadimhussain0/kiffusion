#!/usr/bin/env python3
"""
NASA Image Scraper for Kiffusion.
Downloads high-resolution images from NASA's public API.
"""

import json
import click
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datetime import datetime


NASA_API_BASE = "https://images-api.nasa.gov"


def search_nasa_images(query: str, media_type: str = "image", page: int = 1, page_size: int = 100) -> dict:
    """Search NASA image database."""
    params = {
        "q": query,
        "media_type": media_type,
        "page": page,
        "page_size": page_size,
    }

    response = requests.get(f"{NASA_API_BASE}/search", params=params)
    response.raise_for_status()
    return response.json()


def get_image_urls(item: dict) -> tuple[str, str]:
    """Extract best quality image URL from NASA item."""
    nasa_id = item["data"][0].get("nasa_id", "unknown")

    try:
        response = requests.get(f"{NASA_API_BASE}/asset/{nasa_id}")
        response.raise_for_status()
        assets = response.json()

        urls = [a["href"] for a in assets.get("collection", {}).get("items", [])]

        for suffix in ["~orig.", "~large.", "~medium.", ".jpg", ".png"]:
            for url in urls:
                if suffix in url.lower():
                    return nasa_id, url

        image_urls = [u for u in urls if u.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
        if image_urls:
            return nasa_id, image_urls[0]

    except Exception:
        pass

    preview = item.get("links", [{}])[0].get("href", "")
    return nasa_id, preview


def download_image(args: tuple) -> dict:
    """Download a single image."""
    nasa_id, url, output_dir, metadata = args

    if not url:
        return None

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        ext = ".jpg"
        if ".png" in url.lower():
            ext = ".png"
        elif ".tif" in url.lower():
            ext = ".tif"

        filename = f"{nasa_id}{ext}"
        filepath = output_dir / filename

        with open(filepath, "wb") as f:
            f.write(response.content)

        return {
            "file": filename,
            "nasa_id": nasa_id,
            "title": metadata.get("title", ""),
            "description": metadata.get("description", ""),
            "keywords": metadata.get("keywords", []),
            "center": metadata.get("center", ""),
            "date_created": metadata.get("date_created", ""),
        }

    except Exception:
        return None


@click.command()
@click.option("--query", "-q", multiple=True, required=True,
              help="Search queries (can use multiple)")
@click.option("--output", "-o", default="./datasets/raw/nasa",
              help="Output directory")
@click.option("--limit", "-l", default=50, type=int,
              help="Max images per query")
@click.option("--workers", "-w", default=4, type=int,
              help="Download workers")
def scrape_nasa(query: tuple, output: str, limit: int, workers: int):
    """
    Download images from NASA's public image library.

    Examples:
        python scrape_nasa.py -q "nebula" -q "galaxy" -l 50
        python scrape_nasa.py -q "international space station" -l 30
        python scrape_nasa.py -q "mars surface" -q "jupiter" -l 40
    """
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_items = []

    for q in query:
        click.echo(f"\nSearching for: {q}")

        try:
            results = search_nasa_images(q, page_size=min(limit, 100))
            items = results.get("collection", {}).get("items", [])[:limit]
            click.echo(f"  Found {len(items)} results")
            all_items.extend(items)
        except Exception as e:
            click.echo(f"  Error: {e}")

    if not all_items:
        click.echo("No images found!")
        return

    click.echo(f"\nTotal items to download: {len(all_items)}")

    download_tasks = []
    for item in all_items:
        nasa_id, url = get_image_urls(item)
        metadata = item["data"][0] if item.get("data") else {}
        download_tasks.append((nasa_id, url, output_dir, metadata))

    downloaded = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(
            executor.map(download_image, download_tasks),
            total=len(download_tasks),
            desc="Downloading"
        ))
        downloaded = [r for r in results if r is not None]

    click.echo(f"\nDownloaded {len(downloaded)}/{len(all_items)} images")

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump({
            "source": "NASA Images API",
            "queries": list(query),
            "downloaded_at": datetime.now().isoformat(),
            "images": downloaded,
        }, f, indent=2)

    click.echo(f"Metadata saved to {metadata_file}")
    click.echo("Done!")


if __name__ == "__main__":
    scrape_nasa()
