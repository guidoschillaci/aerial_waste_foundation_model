"""
Download the AerialWaste dataset from Zenodo.

Dataset: https://zenodo.org/records/12607190  (latest version; 7034381 redirects here)
Paper:   Torres & Fraternali, Scientific Data 2023
         https://doi.org/10.1038/s41597-023-01976-9
"""

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

ZENODO_API = "https://zenodo.org/api/records/12607190/files"

DATASET_FILES = [
    {"filename": "training.json", "description": "Training annotations (COCO format)"},
    {"filename": "testing.json",  "description": "Test annotations (COCO format)"},
    {"filename": "images0.zip",   "description": "Images part 0/6"},
    {"filename": "images1.zip",   "description": "Images part 1/6"},
    {"filename": "images2.zip",   "description": "Images part 2/6"},
    {"filename": "images3.zip",   "description": "Images part 3/6"},
    {"filename": "images4.zip",   "description": "Images part 4/6"},
    {"filename": "images5.zip",   "description": "Images part 5/6"},
    {"filename": "images6.zip",   "description": "Images part 6/6"},
]


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return dest

    print(f"  Downloading {desc or dest.name} ...")

    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size * 100 // total_size, 100)
            print(f"\r    {pct}% ", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()
    return dest


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)


def verify_dataset(output_dir: Path) -> None:
    for fname in ["training.json", "testing.json"]:
        fpath = output_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            print(f"  {fname}: {len(data.get('images', []))} images, "
                  f"{len(data.get('annotations', []))} annotations")
        else:
            print(f"  [warning] {fname} not found")

    images_dir = output_dir / "images"
    if images_dir.exists():
        img_count = sum(1 for _ in images_dir.rglob("*") if _.suffix in {".jpg", ".png"})
        print(f"  images/: {img_count} image files")


def main():
    parser = argparse.ArgumentParser(description="Download AerialWaste dataset from Zenodo")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--annotations-only", action="store_true",
                        help="Download only training.json / testing.json")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAerialWaste Dataset Downloader")
    print(f"Source : https://zenodo.org/records/12607190")
    print(f"Output : {output_dir.resolve()}\n")

    for file_info in DATASET_FILES:
        if args.annotations_only and file_info["filename"].endswith(".zip"):
            continue

        url = f"{ZENODO_API}/{file_info['filename']}/content"
        dest = output_dir / file_info["filename"]
        try:
            download_file(url, dest, file_info["description"])
        except Exception as e:
            print(f"  [error] {e}")
            print(f"  Download manually: https://zenodo.org/records/12607190")
            continue

        if dest.suffix == ".zip" and not args.no_extract:
            extract_zip(dest, output_dir)

    print("\nVerifying ...")
    verify_dataset(output_dir)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
