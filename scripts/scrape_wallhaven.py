#!/usr/bin/env python3
"""
Wallhaven Scraper for Kiffusion.
Downloads high-resolution sci-fi and space art from Wallhaven.
API docs: https://wallhaven.cc/help/api
"""

import os
import json
import click
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datetime import datetime
import time


WALLHAVEN_API = "https://wallhaven.cc/api/v1"


def search_wallhaven(
    query: str,
    categories: str = "111",
    purity: str = "100",
    sorting: str = "relevance",
    order: str = "desc",
    page: int = 1,
    api_key: str = None,
) -> dict:
    """Search Wallhaven for images."""
    params = {
        "q": query,
        "categories": categories,
        "purity": purity,
        "sorting": sorting,
        "order": order,
        "atleast": "1920x1080",
        "page": page,
    }

    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    response = requests.get(f"{WALLHAVEN_API}/search", params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def download_image(args: tuple) -> dict:
    """Download a single wallpaper."""
    wallpaper, output_dir, api_key = args

    try:
        wall_id = wallpaper["id"]
        url = wallpaper["path"]

        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key

        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        ext = Path(url).suffix or ".jpg"
        filename = f"wallhaven_{wall_id}{ext}"
        filepath = output_dir / filename

        with open(filepath, "wb") as f:
            f.write(response.content)

        return {
            "file": filename,
            "id": wall_id,
            "url": wallpaper.get("url", ""),
            "resolution": wallpaper.get("resolution", ""),
            "colors": wallpaper.get("colors", []),
            "tags": [t["name"] for t in wallpaper.get("tags", [])],
            "category": wallpaper.get("category", ""),
        }

    except Exception:
        return None


@click.command()
@click.option("--query", "-q", multiple=True, required=True,
              help="Search queries (can use multiple)")
@click.option("--output", "-o", default="./datasets/raw/wallhaven",
              help="Output directory")
@click.option("--limit", "-l", default=100, type=int,
              help="Max images per query")
@click.option("--workers", "-w", default=4, type=int,
              help="Download workers")
@click.option("--api-key", "-k", default=None,
              help="Wallhaven API key (optional, for NSFW content)")
@click.option("--category", "-c", default="general",
              type=click.Choice(["general", "anime", "all"]),
              help="Image category")
def scrape_wallhaven(query: tuple, output: str, limit: int, workers: int, api_key: str, category: str):
    """
    Download high-quality wallpapers from Wallhaven.

    Examples:
        python scrape_wallhaven.py -q "space nebula" -q "sci-fi" -l 100
        python scrape_wallhaven.py -q "cyberpunk city" -q "futuristic" -l 50
        python scrape_wallhaven.py -q "spacecraft" -q "alien planet" -l 100
    """
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = api_key or os.environ.get("WALLHAVEN_API_KEY")

    if category == "general":
        categories = "100"
    elif category == "anime":
        categories = "010"
    else:
        categories = "110"  # General + Anime, skip People

    all_wallpapers = []
    seen_ids = set()

    for q in query:
        click.echo(f"\nSearching for: {q}")

        page = 1
        query_count = 0

        while query_count < limit:
            try:
                results = search_wallhaven(
                    query=q,
                    categories=categories,
                    page=page,
                    api_key=api_key,
                )

                wallpapers = results.get("data", [])
                if not wallpapers:
                    break

                for wall in wallpapers:
                    if query_count >= limit:
                        break
                    if wall["id"] not in seen_ids:
                        seen_ids.add(wall["id"])
                        all_wallpapers.append(wall)
                        query_count += 1

                click.echo(f"  Page {page}: found {len(wallpapers)} wallpapers")

                meta = results.get("meta", {})
                if page >= meta.get("last_page", 1):
                    break

                page += 1
                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                click.echo(f"  Error: {e}")
                break

        click.echo(f"  Total for '{q}': {query_count}")

    if not all_wallpapers:
        click.echo("No wallpapers found!")
        return

    click.echo(f"\nTotal wallpapers to download: {len(all_wallpapers)}")

    download_tasks = [(wall, output_dir, api_key) for wall in all_wallpapers]

    downloaded = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(
            executor.map(download_image, download_tasks),
            total=len(download_tasks),
            desc="Downloading"
        ))
        downloaded = [r for r in results if r is not None]

    click.echo(f"\nDownloaded {len(downloaded)}/{len(all_wallpapers)} wallpapers")

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump({
            "source": "Wallhaven",
            "queries": list(query),
            "downloaded_at": datetime.now().isoformat(),
            "images": downloaded,
        }, f, indent=2)

    click.echo(f"Metadata saved to {metadata_file}")

    all_tags = set()
    for img in downloaded[:20]:
        all_tags.update(img.get("tags", []))
    if all_tags:
        click.echo(f"\nSample tags found: {', '.join(list(all_tags)[:15])}")

    click.echo("Done!")


if __name__ == "__main__":
    scrape_wallhaven()
