#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

MODEL_PATH="${MODEL_PATH:-$QWEN_MIDI_ROOT/runs/pop909-key-chord-melody-to-piano-150m-v1/final-model}"
DATA_DIR="${DATA_DIR:-$QWEN_MIDI_ROOT/data/pop909-key-chord-melody-to-piano-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$QWEN_MIDI_ROOT/runs/key-chord-melody-remi-sampling}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"

python -m pop909_cp.generate_key_chord_melody_to_piano \
  --model-path "$MODEL_PATH" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" \
  --num-samples "$NUM_SAMPLES" --selection-seed 20261016 \
  --generation-seed 20261016 --temperature 1.2 --top-k 5
