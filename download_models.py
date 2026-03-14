#!/usr/bin/env python3
"""
Download pre-trained model files required by people_counter.py
==============================================================

Models downloaded:
  Age estimation  — Levi & Hassner (2015), Caffe
  Gender estimation — Levi & Hassner (2015), Caffe
  YOLOv3          — Redmon & Farhadi (2018), Darknet  [optional, large ~236 MB]

Usage
-----
    # Download age/gender models only (small, ~15 MB total):
    python download_models.py

    # Also download YOLOv3 (better detection accuracy, ~240 MB):
    python download_models.py --yolo
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

_DEFAULT_MODELS_DIR = Path(__file__).parent / "models"

# ---------------------------------------------------------------------------
# File registry: (filename, url, expected_sha256_prefix_or_None)
# ---------------------------------------------------------------------------
AGE_GENDER_FILES = [
    (
        "age_deploy.prototxt",
        "https://raw.githubusercontent.com/spmallick/learnopencv/"
        "master/AgeGender/age_deploy.prototxt",
        None,
    ),
    (
        "gender_deploy.prototxt",
        "https://raw.githubusercontent.com/spmallick/learnopencv/"
        "master/AgeGender/gender_deploy.prototxt",
        None,
    ),
    (
        "age_net.caffemodel",
        "https://github.com/GilLevi/AgeGenderDeepLearning/releases/download/"
        "v0.0/age_net.caffemodel",
        None,
    ),
    (
        "gender_net.caffemodel",
        "https://github.com/GilLevi/AgeGenderDeepLearning/releases/download/"
        "v0.0/gender_net.caffemodel",
        None,
    ),
]

YOLO_FILES = [
    (
        "yolov3.cfg",
        "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        None,
    ),
    (
        "coco.names",
        "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
        None,
    ),
    (
        "yolov3.weights",
        "https://pjreddie.com/media/files/yolov3.weights",
        None,
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded * 100 // total_size, 100)
        bar = "#" * (pct // 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:3d}%  ({downloaded // 1024} KB)")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\r  Downloaded {downloaded // 1024} KB")
        sys.stdout.flush()


def _sha256_prefix(path: Path, length: int = 8) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def download_file(filename: str, url: str, dest_dir: Path, expected_hash=None) -> bool:
    """Download *url* to *dest_dir* / *filename*.

    Skips if the file already exists and passes the hash check.
    Returns True on success, False on failure.
    """
    dest = dest_dir / filename

    if dest.exists():
        if expected_hash and not _sha256_prefix(dest).startswith(expected_hash):
            print(f"  [WARN] {filename} exists but hash mismatch — re-downloading.")
        else:
            print(f"  [SKIP] {filename} already present.")
            return True

    print(f"\n  Downloading {filename}")
    print(f"  Source: {url}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
        print()  # newline after progress bar
    except Exception as exc:
        print(f"\n  [ERROR] Failed to download {filename}: {exc}")
        if dest.exists():
            dest.unlink()
        return False

    if expected_hash:
        actual = _sha256_prefix(dest)
        if not actual.startswith(expected_hash):
            print(f"  [ERROR] Hash mismatch for {filename}: got {actual}, expected {expected_hash}")
            dest.unlink()
            return False

    size_kb = dest.stat().st_size // 1024
    print(f"  [OK]   {filename} saved ({size_kb} KB)")
    return True


def download_batch(files: list, label: str, dest_dir: Path) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    success = True
    for filename, url, expected_hash in files:
        if not download_file(filename, url, dest_dir, expected_hash):
            success = False
    return success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download model files for people_counter.py"
    )
    ap.add_argument(
        "--yolo",
        action="store_true",
        help="Also download YOLOv3 weights (~236 MB, optional but recommended).",
    )
    ap.add_argument(
        "--models-dir",
        default=str(_DEFAULT_MODELS_DIR),
        help=f"Destination directory (default: {_DEFAULT_MODELS_DIR})",
    )
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    ok = download_batch(AGE_GENDER_FILES, "Age & Gender estimation models", models_dir)

    if args.yolo:
        yolo_ok = download_batch(YOLO_FILES, "YOLOv3 object detection model", models_dir)
        ok = ok and yolo_ok

    print(f"\n{'='*60}")
    if ok:
        print("  All downloads completed successfully.")
    else:
        print("  Some downloads FAILED. Check the messages above.")
        print("  people_counter.py will fall back to the built-in HOG detector")
        print("  and skip age/gender estimation for missing models.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
