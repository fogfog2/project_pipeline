#!/usr/bin/env bash
set -euo pipefail

# Run inside a clean MMDetection 3.3.0 environment. This script deliberately
# downloads into a user-selected directory; it never changes the registry.
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <destination-directory>" >&2
  exit 2
fi

destination=$1
mkdir -p "$destination"

python -m pip install "openmim>=0.3" "mmdet==3.3.0"
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest "$destination/rtmdet-tiny"
mim download mmdet --config yolox_s_8xb8-300e_coco --dest "$destination/yolox-s"

echo "Downloaded model-zoo artifacts to $destination. Register the matching config and .pth files as separate model bundles."
