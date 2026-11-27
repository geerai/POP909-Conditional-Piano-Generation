#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

MODEL_PATH="${MODEL_PATH:-$QWEN_MIDI_ROOT/runs/unconditional-piano-pop-909/final-model}"
DATA_DIR="${DATA_DIR:-$QWEN_MIDI_ROOT/data/pop909-melody-to-piano}"
OUTPUT_DIR="${OUTPUT_DIR:-$QWEN_MIDI_ROOT/runs/pop909-melody-to-piano-v1}"

torchrun --standalone --nproc_per_node=2 \
  -m pop909_cp.train_melody_to_piano \
  --model-path "$MODEL_PATH" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" \
  --mode train --epochs 20 --learning-rate 1e-4 --warmup-ratio 0.05 \
  --per-device-batch-size 4 --gradient-accumulation-steps 4 \
  --max-grad-norm 1.0 --early-stopping-patience 4
