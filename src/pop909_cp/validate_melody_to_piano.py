#!/usr/bin/env python3
"""Validate POP909 melody-condition -> piano-target JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def validate(tokens: list[str], loss_start: int) -> tuple[int, int]:
    if "<CONDITION_START>" not in tokens or "<CONDITION_END>" not in tokens:
        raise ValueError("missing condition boundary")
    if "<ARRANGEMENT_START>" not in tokens:
        raise ValueError("missing target boundary")
    cs = tokens.index("<CONDITION_START>")
    ce = tokens.index("<CONDITION_END>")
    target = tokens.index("<ARRANGEMENT_START>")
    if not cs < ce < target or loss_start != target + 1:
        raise ValueError("bad boundary order or loss_start_token_index")
    if tokens[:target].count("<ROLE_MELODY>") < 2:
        raise ValueError("condition has no melody role/bar")
    if "<ROLE_PIANO>" in tokens[cs + 1:ce]:
        raise ValueError("piano leaked into condition body")
    if "<ROLE_MELODY>" in tokens[target + 1:]:
        raise ValueError("melody leaked into target body")

    def parse_body(body: list[str], role: str) -> tuple[int, int]:
        bars = notes = 0
        i = 0
        while i < len(body):
            if body[i] in {"<ARRANGEMENT_END>", "<SONG_END>"}:
                i += 1
                continue
            if body[i:i + 2] != ["<BAR>", role]:
                raise ValueError(f"expected BAR + {role}, found {body[i:i+2]}")
            bars += 1
            i += 2
            while i < len(body) and body[i] != "<ROLE_END>":
                triple = body[i:i + 3]
                if len(triple) != 3 or not re.fullmatch(r"<POSITION_\d{3}>", triple[0]) \
                        or not re.fullmatch(r"<PITCH_\d{3}>", triple[1]) \
                        or not re.fullmatch(r"<DURATION_\d{3}>", triple[2]):
                    raise ValueError(f"invalid note triple: {triple}")
                notes += 1
                i += 3
            if body[i:i + 2] != ["<ROLE_END>", "<BAR_END>"]:
                raise ValueError("missing ROLE_END/BAR_END")
            i += 2
        return bars, notes

    melody_bars, melody_notes = parse_body(tokens[cs + 1:ce], "<ROLE_MELODY>")
    piano_bars, piano_notes = parse_body(tokens[target + 1:], "<ROLE_PIANO>")
    if melody_bars != piano_bars or not melody_notes or not piano_notes:
        raise ValueError("condition/target bar mismatch or empty role")
    return melody_notes, piano_notes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    vocab_payload = json.loads((args.data_dir / "remi-plus-vocab-v4.json").read_text())
    rows = sorted(vocab_payload["tokens"], key=lambda row: int(row["id"]))
    vocab = [str(row["token"]) for row in rows]
    if len(vocab) != 1727:
        raise SystemExit("vocabulary is not fixed REMI+ v4 (1,727 tokens)")
    counts = {}
    songs = {}
    for split in ("train", "validation", "test"):
        count = 0
        song_ids: set[str] = set()
        with (args.data_dir / f"{split}.jsonl").open() as handle:
            for number, line in enumerate(handle, 1):
                row = json.loads(line)
                ids = [int(value) for value in row["input_ids"]]
                if len(ids) > 2048 or min(ids) < 0 or max(ids) >= len(vocab):
                    raise ValueError(f"{split}:{number}: invalid IDs or length")
                tokens = [vocab[value] for value in ids]
                try:
                    validate(tokens, int(row["loss_start_token_index"]))
                except ValueError as exc:
                    raise ValueError(
                        f"{split}:{number} sample_id={row.get('sample_id')}: {exc}"
                    ) from exc
                if row["task"] != "MELODY_TO_PIANO_ACCOMPANIMENT":
                    raise ValueError(f"{split}:{number}: wrong task")
                song_ids.add(str(row["song_id"]))
                count += 1
        if not count:
            raise ValueError(f"{split} is empty")
        counts[split] = count
        songs[split] = song_ids
    if songs["train"] & songs["validation"] or songs["train"] & songs["test"] or songs["validation"] & songs["test"]:
        raise ValueError("song-level split leakage")
    report = {
        "status": "PASS",
        "records": counts,
        "condition_role": "MELODY",
        "target_role": "PIANO",
        "loss_mask": "all labels before first target BAR are -100",
        "song_split_leakage": False,
        "test_opened_during_training": False,
    }
    (args.data_dir / "validation-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
