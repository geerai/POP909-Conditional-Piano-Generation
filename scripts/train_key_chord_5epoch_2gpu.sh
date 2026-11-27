#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

MODEL_PATH="${MODEL_PATH:-$QWEN_MIDI_ROOT/runs/pop909-key-chord-melody-to-piano-150m-initial}"
DATA_DIR="${DATA_DIR:-$QWEN_MIDI_ROOT/data/pop909-key-chord-melody-to-piano-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$QWEN_MIDI_ROOT/runs/pop909-key-chord-melody-to-piano-150m-v1}"

torchrun --standalone --nproc_per_node=2 \
  -m pop909_cp.train_key_chord_melody_to_piano \
  --model-path "$MODEL_PATH" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" \
  --mode train --epochs 5 --learning-rate 1e-5 --warmup-ratio 0.10 \
  --per-device-batch-size 2 --gradient-accumulation-steps 8 \
  --max-grad-norm 1.0
