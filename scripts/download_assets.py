#!/usr/bin/env python3
"""
Download supporting assets:
  - CMU Pronouncing Dictionary  →  data/cmudict.dict
  - LibTorch (CPU) archive hint

Usage:
    python scripts/download_assets.py [--data-dir data]
"""

import argparse
import os
import urllib.request
import sys


CMU_DICT_URL = (
    "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
)


def download(url: str, dest: str) -> None:
    print(f"Downloading {url}")
    print(f"  → {dest}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    size = os.path.getsize(dest)
    print(f"  ✓ {size // 1024} KB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    cmu_path = os.path.join(args.data_dir, "cmudict.dict")
    if os.path.exists(cmu_path):
        print(f"CMU dict already exists at {cmu_path}, skipping.")
    else:
        download(CMU_DICT_URL, cmu_path)

    libtorch_version = "2.5.1"
    print(f"""
Assets downloaded.

LibTorch setup (if not already installed):
  1. Download LibTorch {libtorch_version} CPU:
     https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-{libtorch_version}%2Bcpu.zip

  2. Extract and note the path, then configure cmake:
     cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch

  Or if you have PyTorch installed in a venv:
     cmake .. -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
""")


if __name__ == "__main__":
    main()
