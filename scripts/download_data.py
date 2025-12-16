#!/usr/bin/env python3
"""Download WebQSP and CWQ datasets for LightPROF.

This script downloads the required datasets for KGQA training and evaluation:
- WebQSP: 4,737 questions with SPARQL parses
- CWQ: 34,689 complex compositional questions
- Freebase subset: Knowledge graph triples

Usage:
    python scripts/download_data.py --dataset webqsp
    python scripts/download_data.py --dataset cwq
    python scripts/download_data.py --dataset all
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# Data directory structure
DATA_DIR = Path("data")

# Dataset URLs and checksums
DATASETS = {
    "webqsp": {
        "description": "WebQuestions Semantic Parses (WebQSP)",
        "files": {
            "train": {
                "url": "https://raw.githubusercontent.com/salesforce/rng-kbqa/main/data/WebQSP/WebQSP.train.json",
                "filename": "webqsp/train.json",
            },
            "test": {
                "url": "https://raw.githubusercontent.com/salesforce/rng-kbqa/main/data/WebQSP/WebQSP.test.json",
                "filename": "webqsp/test.json",
            },
        },
    },
    "cwq": {
        "description": "ComplexWebQuestions (CWQ)",
        "files": {
            "train": {
                "url": "https://raw.githubusercontent.com/salesforce/rng-kbqa/main/data/CWQ/ComplexWebQuestions_train.json",
                "filename": "cwq/train.json",
            },
            "dev": {
                "url": "https://raw.githubusercontent.com/salesforce/rng-kbqa/main/data/CWQ/ComplexWebQuestions_dev.json",
                "filename": "cwq/dev.json",
            },
            "test": {
                "url": "https://raw.githubusercontent.com/salesforce/rng-kbqa/main/data/CWQ/ComplexWebQuestions_test.json",
                "filename": "cwq/test.json",
            },
        },
    },
    "freebase": {
        "description": "Freebase subset (FB15k-237 or custom)",
        "files": {
            "triples": {
                "url": "https://raw.githubusercontent.com/TimDettmers/ConvE/master/data/FB15k-237/train.txt",
                "filename": "freebase_triples.tsv",
            },
        },
        "note": "For full Freebase, use scripts/convert_kg.py with a larger dump.",
    },
}


def download_file(url: str, dest: Path, description: str) -> bool:
    """Download a file from URL to destination.

    Args:
        url: Source URL.
        dest: Destination path.
        description: Human-readable description for logging.

    Returns:
        True if download succeeded, False otherwise.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"  [SKIP] {description}: already exists at {dest}")
        return True

    print(f"  [DOWNLOAD] {description}")
    print(f"    URL: {url}")
    print(f"    Destination: {dest}")

    try:
        # Try with urllib first
        with urllib.request.urlopen(url, timeout=60) as response:
            with open(dest, "wb") as f:
                shutil.copyfileobj(response, f)
        print(f"  [OK] Downloaded {dest.stat().st_size / 1024:.1f} KB")
        return True
    except urllib.error.URLError as e:
        print(f"  [ERROR] Failed to download: {e}")
        return False


def download_with_curl(url: str, dest: Path, description: str) -> bool:
    """Download using curl as fallback.

    Args:
        url: Source URL.
        dest: Destination path.
        description: Human-readable description for logging.

    Returns:
        True if download succeeded, False otherwise.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"  [SKIP] {description}: already exists at {dest}")
        return True

    print(f"  [DOWNLOAD] {description} (using curl)")
    print(f"    URL: {url}")

    try:
        result = subprocess.run(
            ["curl", "-sL", "-o", str(dest), url],
            check=True,
            capture_output=True,
            text=True,
        )
        if dest.exists():
            print(f"  [OK] Downloaded {dest.stat().st_size / 1024:.1f} KB")
            return True
        return False
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [ERROR] curl failed: {e}")
        return False


def download_dataset(name: str) -> bool:
    """Download a specific dataset.

    Args:
        name: Dataset name ('webqsp', 'cwq', or 'freebase').

    Returns:
        True if all files downloaded successfully.
    """
    if name not in DATASETS:
        print(f"[ERROR] Unknown dataset: {name}")
        print(f"  Available: {', '.join(DATASETS.keys())}")
        return False

    dataset = DATASETS[name]
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset['description']}")
    print(f"{'='*60}")

    if "note" in dataset:
        print(f"Note: {dataset['note']}")

    success = True
    for file_key, file_info in dataset["files"].items():
        url = file_info["url"]
        dest = DATA_DIR / file_info["filename"]

        # Try urllib first, then curl as fallback
        if not download_file(url, dest, f"{name}/{file_key}"):
            if not download_with_curl(url, dest, f"{name}/{file_key}"):
                success = False

    return success


def verify_dataset(name: str) -> bool:
    """Verify that dataset files exist.

    Args:
        name: Dataset name.

    Returns:
        True if all expected files exist.
    """
    if name not in DATASETS:
        return False

    dataset = DATASETS[name]
    all_exist = True

    for file_info in dataset["files"].values():
        path = DATA_DIR / file_info["filename"]
        if not path.exists():
            print(f"  [MISSING] {path}")
            all_exist = False
        else:
            print(f"  [OK] {path} ({path.stat().st_size / 1024:.1f} KB)")

    return all_exist


def print_usage() -> None:
    """Print usage information."""
    print("\nAvailable datasets:")
    for name, info in DATASETS.items():
        print(f"  {name:12} - {info['description']}")

    print("\nData will be downloaded to:")
    print(f"  {DATA_DIR.absolute()}")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Download datasets for LightPROF KGQA training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_data.py --dataset webqsp
    python scripts/download_data.py --dataset cwq
    python scripts/download_data.py --dataset all
    python scripts/download_data.py --verify webqsp
        """,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        choices=["webqsp", "cwq", "freebase", "all"],
        help="Dataset to download",
    )
    parser.add_argument(
        "--verify",
        "-v",
        choices=["webqsp", "cwq", "freebase", "all"],
        help="Verify dataset files exist",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DATA_DIR,
        help=f"Output directory (default: {DATA_DIR})",
    )

    args = parser.parse_args()

    global DATA_DIR
    DATA_DIR = args.output_dir

    if args.verify:
        print("Verifying datasets...")
        datasets = list(DATASETS.keys()) if args.verify == "all" else [args.verify]

        all_ok = True
        for name in datasets:
            print(f"\n{name}:")
            if not verify_dataset(name):
                all_ok = False

        return 0 if all_ok else 1

    if args.dataset:
        datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

        all_ok = True
        for name in datasets:
            if not download_dataset(name):
                all_ok = False

        print("\n" + "=" * 60)
        if all_ok:
            print("All downloads completed successfully!")
        else:
            print("Some downloads failed. Check the errors above.")
        print("=" * 60)

        return 0 if all_ok else 1

    # No arguments - print help
    print("LightPROF Data Downloader")
    print_usage()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
