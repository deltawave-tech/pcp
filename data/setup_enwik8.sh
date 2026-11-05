#!/bin/bash
# Setup script for enwik8 dataset (100MB Wikipedia dump)
# This script downloads and prepares the enwik8 dataset for NanoGPT medium experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Downloading enwik8 dataset..."
wget -q --show-progress http://mattmahoney.net/dc/enwik8.zip

echo "Extracting..."
python3 -c "import zipfile; zipfile.ZipFile('enwik8.zip').extractall()"

echo "Renaming to .txt extension..."
mv enwik8 enwik8.txt

echo "Cleaning up..."
rm enwik8.zip

echo "✓ enwik8.txt is ready ($(du -h enwik8.txt | cut -f1))"
