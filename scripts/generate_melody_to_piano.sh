#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

MODEL_PATH="${MODEL_PATH:-$QWEN_MIDI_ROOT/runs/pop909-melody-to-piano-v1/final-model}"
DATA_DIR="${DATA_DIR:-$QWEN_MIDI_ROOT/data/pop909-melody-to-piano}"
OUTPUT_DIR="${OUTPUT_DIR:-$QWEN_MIDI_ROOT/runs/pop909-melody-to-piano-generation20}"

python -m pop909_cp.generate_melody_to_piano \
  --model-path "$MODEL_PATH" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" \
  --sample-count 20 --top-k 20 --top-p 0.95 --temperature 1.0 \
  --seed 20261006
