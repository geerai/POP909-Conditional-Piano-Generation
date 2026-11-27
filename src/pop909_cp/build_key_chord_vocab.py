#!/usr/bin/env python3
"""Append the key/chord tokens while preserving every base vocabulary ID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PITCH_CLASS_NAMES = (
    "C", "C_SHARP", "D", "D_SHARP", "E", "F",
    "F_SHARP", "G", "G_SHARP", "A", "A_SHARP", "B",
)
CHORD_QUALITIES = (
    "M", "m", "o", "aug", "sus2", "sus4", "D7", "M7", "m7",
    "half_o7", "o7", "mM7", "aug7",
)


def build_extended_vocab(source: Path) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = sorted(payload["tokens"], key=lambda row: int(row["id"]))
    if [int(row["id"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Base vocabulary IDs are not contiguous")
    old_size = len(rows)
    existing = {row["token"] for row in rows}

    def append(token: str, family: str) -> None:
        if token in existing:
            raise ValueError(f"Token already exists: {token}")
        rows.append({"id": len(rows), "token": token, "family": family})
        existing.add(token)

    append("<TASK_KEY_CHORD_MELODY_TO_PIANO>", "TASK")
    append("<ROLE_CHORD>", "ROLE")
    for index in range(32):
        required = f"<BAR_{index:03d}>"
        if required not in existing:
            raise ValueError(f"Base vocabulary lacks {required}")
    for name in PITCH_CLASS_NAMES:
        append(f"<KEY_ROOT_{name}>", "KEY_ROOT")
    for mode in ("MAJOR", "MINOR"):
        append(f"<KEY_MODE_{mode}>", "KEY_MODE")
    for name in PITCH_CLASS_NAMES:
        append(f"<CHORD_ROOT_{name}>", "CHORD_ROOT")
    for quality in CHORD_QUALITIES:
        append(f"<CHORD_QUALITY_{quality}>", "CHORD_QUALITY")
    for name in PITCH_CLASS_NAMES:
        append(f"<CHORD_BASS_{name}>", "CHORD_BASS")
    append("<CHORD_OTHER>", "CHORD")
    append("<NO_CHORD>", "CHORD")

    result = dict(payload)
    result.update(
        {
            "version": "remi-plus-v4-key-chord-v1",
            "vocab_size": len(rows),
            "base_vocab_size": old_size,
            "base_vocab_ids_preserved": True,
            "tokens": rows,
            "key_chord_added_tokens": [row["token"] for row in rows[old_size:]],
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-vocab", type=Path, required=True)
    parser.add_argument("--output-vocab", type=Path, required=True)
    args = parser.parse_args()

    payload = build_extended_vocab(args.base_vocab)
    args.output_vocab.parent.mkdir(parents=True, exist_ok=True)
    args.output_vocab.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output_vocab),
                "base_vocab_size": payload["base_vocab_size"],
                "extended_vocab_size": payload["vocab_size"],
                "old_ids_preserved": payload["base_vocab_ids_preserved"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
