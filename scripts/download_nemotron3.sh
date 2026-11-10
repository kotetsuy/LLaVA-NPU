#!/usr/bin/env bash
set -euo pipefail

repo='https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF/resolve/main'
dest="${HOME}/nemotron-3"
mkdir -p "$dest"

for file in NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf mmproj-F16.gguf; do
  if [[ -f "$dest/$file" ]]; then
    printf 'Already exists, skipping: %s\n' "$dest/$file"
    continue
  fi
  printf 'Downloading %s\n' "$file"
  curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
    --output "$dest/$file.part" "$repo/$file?download=true"
  mv -- "$dest/$file.part" "$dest/$file"
done
printf 'Downloads complete in %s\n' "$dest"
