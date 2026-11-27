#!/usr/bin/env python3
"""Build POP909 Key+Chord+Melody -> Piano conditional-LM data.

Inputs:
  * Original POP909: independent MELODY and PIANO tracks.
  * POP909-CL: corrected chord track, key signatures, meters, and tempo.
  * accepted-songs.json: output of audit_pop909_cl.py.

The two MIDI sources are converted independently from ticks to 48 units per
quarter note. This is required because a small number of original files use
960 PPQ while their POP909-CL counterparts use 480 PPQ.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import miditoolkit
except ImportError as exc:
    raise SystemExit("Install miditoolkit in the qwen-midi environment.") from exc

try:
    from .prepare_pop909_melody_to_piano import (
        GRID_PER_QUARTER,
        MAX_DURATION,
        MAX_TOKENS,
        PianoNote,
        beat_alignment,
        canonical_track_name,
        clamp_tempo,
        load_index_metadata,
        normalize_quantized_notes,
        split_map,
        tick_to_grid,
        write_json,
    )
except ImportError:  # Support direct execution from this directory.
    from prepare_pop909_melody_to_piano import (
        GRID_PER_QUARTER,
        MAX_DURATION,
        MAX_TOKENS,
        PianoNote,
        beat_alignment,
        canonical_track_name,
        clamp_tempo,
        load_index_metadata,
        normalize_quantized_notes,
        split_map,
        tick_to_grid,
        write_json,
    )


TASK_TOKEN = "<TASK_KEY_CHORD_MELODY_TO_PIANO>"
MAX_BARS_PER_WINDOW = 32
PITCH_CLASS_NAMES = ("C", "C_SHARP", "D", "D_SHARP", "E", "F", "F_SHARP", "G", "G_SHARP", "A", "A_SHARP", "B")
TRIAD_NAMES = ("M", "m", "o", "aug", "sus2", "sus4")
TRIAD_DEGREES = (
    frozenset((0, 4, 7)),
    frozenset((0, 3, 7)),
    frozenset((0, 3, 6)),
    frozenset((0, 4, 8)),
    frozenset((0, 2, 7)),
    frozenset((0, 5, 7)),
)
SEVENTH_NAMES = ("D7", "M7", "m7", "half_o7", "o7", "mM7", "aug7")
SEVENTH_DEGREES = (
    frozenset((0, 4, 7, 10)),
    frozenset((0, 4, 7, 11)),
    frozenset((0, 3, 7, 10)),
    frozenset((0, 3, 6, 10)),
    frozenset((0, 3, 6, 9)),
    frozenset((0, 3, 7, 11)),
    frozenset((0, 4, 8, 10)),
)
ROOT_TO_PC = {
    "C": 0, "B#": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4, "FB": 4,
    "E#": 5, "F": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11, "CB": 11,
}


@dataclass(frozen=True)
class Bar:
    index: int
    start: int
    end: int
    meter: tuple[int, int]


@dataclass(frozen=True)
class ChordState:
    root: int | None
    quality: str | None
    bass: int | None
    other: bool = False


NO_CHORD = ChordState(None, None, None)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Required file not found: {path}") from exc


def append_vocab_token(rows: list[dict[str, Any]], token: str, family: str) -> None:
    if any(row["token"] == token for row in rows):
        raise ValueError(f"Token already exists: {token}")
    rows.append({"id": len(rows), "token": token, "family": family})


def build_extended_vocab(source: Path) -> tuple[dict[str, int], list[str], dict[str, Any]]:
    payload = read_json(source)
    rows = sorted(payload["tokens"], key=lambda row: int(row["id"]))
    if [int(row["id"]) for row in rows] != list(range(len(rows))):
        raise SystemExit(f"Vocabulary IDs are not contiguous: {source}")
    old_size = len(rows)
    append_vocab_token(rows, TASK_TOKEN, "TASK")
    append_vocab_token(rows, "<ROLE_CHORD>", "ROLE")
    # REMI+ v4 already contains <BAR_000> through <BAR_127>. We reuse the
    # first 32 as relative in-window bar IDs, preserving every old ID.
    for index in range(MAX_BARS_PER_WINDOW):
        if not any(row["token"] == f"<BAR_{index:03d}>" for row in rows):
            raise ValueError(f"Base vocabulary lacks <BAR_{index:03d}>")
    for name in PITCH_CLASS_NAMES:
        append_vocab_token(rows, f"<KEY_ROOT_{name}>", "KEY_ROOT")
    for mode in ("MAJOR", "MINOR"):
        append_vocab_token(rows, f"<KEY_MODE_{mode}>", "KEY_MODE")
    for name in PITCH_CLASS_NAMES:
        append_vocab_token(rows, f"<CHORD_ROOT_{name}>", "CHORD_ROOT")
    for quality in TRIAD_NAMES + SEVENTH_NAMES:
        append_vocab_token(rows, f"<CHORD_QUALITY_{quality}>", "CHORD_QUALITY")
    for name in PITCH_CLASS_NAMES:
        append_vocab_token(rows, f"<CHORD_BASS_{name}>", "CHORD_BASS")
    append_vocab_token(rows, "<CHORD_OTHER>", "CHORD")
    append_vocab_token(rows, "<NO_CHORD>", "CHORD")
    result = dict(payload)
    result.update({
        "version": "remi-plus-v4-key-chord-v1",
        "vocab_size": len(rows),
        "base_vocab_size": old_size,
        "base_vocab_ids_preserved": True,
        "tokens": rows,
        "key_chord_added_tokens": [row["token"] for row in rows[old_size:]],
    })
    tokens = [str(row["token"]) for row in rows]
    return dict(zip(tokens, range(len(tokens)))), tokens, result


def discover_cl_midis(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted((root / "POP909_processed").glob("*.mid")):
        match = re.fullmatch(r"\s*(\d{3})\s*", path.stem)
        if match:
            result[match.group(1)] = path
    return result


def pc_token(prefix: str, pc: int) -> str:
    return f"<{prefix}_{PITCH_CLASS_NAMES[pc % 12]}>"


def parse_key_name(value: str) -> tuple[int, str]:
    text = str(value).strip().replace("♭", "b").replace("♯", "#")
    lower = text.lower()
    mode = "MINOR" if lower.endswith("m") or "minor" in lower else "MAJOR"
    root = re.split(r"\s+", text)[0]
    if root.lower().endswith("m"):
        root = root[:-1]
    root = root.upper().replace("B", "B")
    # Preserve a lowercase flat marker after uppercasing.
    root = root[0] + ("b" if len(root) > 1 and root[1] in {"B", "b"} else "#" if len(root) > 1 and root[1] == "#" else "")
    pc = ROOT_TO_PC.get(root.upper())
    if pc is None:
        raise ValueError(f"Unsupported key signature: {value!r}")
    return pc, mode


def chord_from_notes(notes: Sequence[Any]) -> ChordState:
    pcs = frozenset(int(note.pitch) % 12 for note in notes)
    if not pcs:
        return NO_CHORD
    bass = min(notes, key=lambda note: int(note.pitch)).pitch % 12
    for root in sorted(pcs):
        degrees = frozenset((pc - root) % 12 for pc in pcs)
        for quality, pattern in zip(SEVENTH_NAMES, SEVENTH_DEGREES):
            if degrees == pattern:
                return ChordState(root, quality, bass)
        for quality, pattern in zip(TRIAD_NAMES, TRIAD_DEGREES):
            if degrees == pattern:
                return ChordState(root, quality, bass)
    return ChordState(None, None, bass, other=True)


def chord_state_tokens(state: ChordState) -> list[str]:
    if state == NO_CHORD:
        return ["<NO_CHORD>"]
    if state.other:
        return ["<CHORD_OTHER>"]
    assert state.root is not None and state.quality is not None and state.bass is not None
    return [
        pc_token("CHORD_ROOT", state.root),
        f"<CHORD_QUALITY_{state.quality}>",
        pc_token("CHORD_BASS", state.bass),
    ]


def make_chord_changes(midi: Any) -> list[tuple[int, ChordState]]:
    if len(midi.instruments) < 2:
        raise ValueError("POP909-CL has no chord track")
    ppq = int(midi.ticks_per_beat)
    grouped: dict[int, list[Any]] = {}
    for note in midi.instruments[-1].notes:
        grouped.setdefault(int(note.start), []).append(note)
    blocks: list[tuple[int, int, ChordState]] = []
    for start_tick, notes in sorted(grouped.items()):
        start = tick_to_grid(start_tick, ppq)
        end = max(start + 1, tick_to_grid(max(int(note.end) for note in notes), ppq))
        blocks.append((start, end, chord_from_notes(notes)))
    changes: dict[int, ChordState] = {0: NO_CHORD}
    for index, (start, end, state) in enumerate(blocks):
        changes[start] = state
        next_start = blocks[index + 1][0] if index + 1 < len(blocks) else None
        if next_start is not None and next_start > end:
            changes[end] = NO_CHORD
    return sorted(changes.items())


def make_key_changes(midi: Any) -> list[tuple[int, tuple[int, str]]]:
    ppq = int(midi.ticks_per_beat)
    changes = [
        (tick_to_grid(int(event.time), ppq), parse_key_name(event.key_name))
        for event in sorted(midi.key_signature_changes, key=lambda event: int(event.time))
    ]
    if not changes:
        raise ValueError("POP909-CL has no key signature")
    # A start-beat correction can move the first key meta-event after tick 0.
    # The first declared key is still the only defensible active key for the
    # pickup preceding that event, so explicitly carry it back to grid zero.
    if changes[0][0] > 0:
        changes.insert(0, (0, changes[0][1]))
    return changes


def meter_at_zero(midi: Any) -> tuple[int, int]:
    active = (4, 4)
    for event in sorted(midi.time_signature_changes, key=lambda event: int(event.time)):
        if int(event.time) > 0:
            break
        active = (int(event.numerator), int(event.denominator))
    return active


def meter_size(meter: tuple[int, int]) -> int:
    numerator, denominator = meter
    raw = numerator * 4 * GRID_PER_QUARTER
    if denominator <= 0 or raw % denominator:
        raise ValueError(f"Meter cannot be represented on grid: {meter}")
    return raw // denominator


def make_bars(midi: Any, maximum_grid: int) -> list[Bar]:
    ppq = int(midi.ticks_per_beat)
    changes: dict[int, tuple[int, int]] = {0: (4, 4)}
    for event in midi.time_signature_changes:
        changes[tick_to_grid(int(event.time), ppq)] = (
            int(event.numerator), int(event.denominator)
        )
    ordered = sorted(changes.items())
    position = 0
    meter = ordered[0][1]
    change_index = 1
    bars: list[Bar] = []
    while position <= maximum_grid:
        while change_index < len(ordered) and ordered[change_index][0] == position:
            meter = ordered[change_index][1]
            change_index += 1
        end = position + meter_size(meter)
        if change_index < len(ordered) and position < ordered[change_index][0] < end:
            raise ValueError(
                f"Time-signature change at grid {ordered[change_index][0]} is not on a bar boundary"
            )
        bars.append(Bar(len(bars), position, end, meter))
        position = end
    return bars


def active_value(changes: Sequence[tuple[int, Any]], position: int) -> Any:
    times = [row[0] for row in changes]
    index = bisect.bisect_right(times, position) - 1
    if index < 0:
        raise ValueError(f"No active event at grid {position}")
    return changes[index][1]


def assign_notes_to_bars(notes: Sequence[PianoNote], bars: Sequence[Bar]) -> list[list[PianoNote]]:
    starts = [bar.start for bar in bars]
    result: list[list[PianoNote]] = [[] for _ in bars]
    for note in notes:
        index = bisect.bisect_right(starts, note.start) - 1
        if index >= 0 and note.start < bars[index].end:
            duration = min(MAX_DURATION, note.end - note.start)
            if duration > 0:
                result[index].append(PianoNote(note.start, note.start + duration, note.pitch, note.velocity))
    for row in result:
        row.sort(key=lambda note: (note.start, note.pitch, note.end))
    return result


def serialize_note_role(role: str, notes: Sequence[PianoNote], bar: Bar) -> list[str]:
    tokens = [f"<ROLE_{role}>"]
    for note in notes:
        tokens.extend([
            f"<POSITION_{note.start - bar.start:03d}>",
            f"<PITCH_{note.pitch:03d}>",
            f"<DURATION_{note.end - note.start:03d}>",
        ])
    tokens.append("<ROLE_END>")
    return tokens


def serialize_condition_bar(
    relative_index: int,
    bar: Bar,
    key_changes: Sequence[tuple[int, tuple[int, str]]],
    chord_changes: Sequence[tuple[int, ChordState]],
    melody: Sequence[PianoNote],
) -> list[str]:
    root, mode = active_value(key_changes, bar.start)
    tokens = [
        f"<BAR_{relative_index:03d}>",
        pc_token("KEY_ROOT", root),
        f"<KEY_MODE_{mode}>",
        "<ROLE_CHORD>",
        "<POSITION_000>",
    ]
    tokens.extend(chord_state_tokens(active_value(chord_changes, bar.start)))
    for time, state in chord_changes:
        if bar.start < time < bar.end:
            tokens.append(f"<POSITION_{time - bar.start:03d}>")
            tokens.extend(chord_state_tokens(state))
    tokens.append("<ROLE_END>")
    tokens.extend(serialize_note_role("MELODY", melody, bar))
    tokens.append("<BAR_END>")
    return tokens


def serialize_window(
    bars: Sequence[Bar],
    melody_bars: Sequence[Sequence[PianoNote]],
    piano_bars: Sequence[Sequence[PianoNote]],
    key_changes: Sequence[tuple[int, tuple[int, str]]],
    chord_changes: Sequence[tuple[int, ChordState]],
    tempo: int,
    start: int,
    end: int,
    true_song_start: bool,
    true_song_end: bool,
) -> tuple[list[str], int]:
    meter = bars[start].meter
    if any(bar.meter != meter for bar in bars[start : end + 1]):
        raise ValueError("A training window cannot cross a meter change")
    tokens: list[str] = []
    if true_song_start:
        tokens.append("<SONG_START>")
    tokens.extend([
        TASK_TOKEN,
        f"<METER_{meter[0]}_{meter[1]}>",
        f"<TEMPO_{tempo:03d}>",
        "<CONDITION_START>",
    ])
    for relative, absolute in enumerate(range(start, end + 1)):
        tokens.extend(serialize_condition_bar(
            relative, bars[absolute], key_changes, chord_changes, melody_bars[absolute]
        ))
    tokens.extend(["<CONDITION_END>", "<ARRANGEMENT_START>"])
    loss_start = len(tokens)
    for relative, absolute in enumerate(range(start, end + 1)):
        tokens.append(f"<BAR_{relative:03d}>")
        tokens.extend(serialize_note_role("PIANO", piano_bars[absolute], bars[absolute]))
        tokens.append("<BAR_END>")
    if true_song_end:
        tokens.extend(["<ARRANGEMENT_END>", "<SONG_END>"])
    return tokens, loss_start


def make_windows(
    bars: Sequence[Bar],
    melody_bars: Sequence[Sequence[PianoNote]],
    piano_bars: Sequence[Sequence[PianoNote]],
    key_changes: Sequence[tuple[int, tuple[int, str]]],
    chord_changes: Sequence[tuple[int, ChordState]],
    tempo: int,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    start = 0
    while start < len(bars):
        accepted: tuple[int, list[str], int] | None = None
        meter = bars[start].meter
        maximum_end = min(len(bars) - 1, start + MAX_BARS_PER_WINDOW - 1)
        for end in range(start, maximum_end + 1):
            if bars[end].meter != meter:
                break
            tokens, loss_start = serialize_window(
                bars, melody_bars, piano_bars, key_changes, chord_changes, tempo,
                start, end, start == 0, end == len(bars) - 1,
            )
            if len(tokens) > MAX_TOKENS:
                break
            accepted = end, tokens, loss_start
        if accepted is None:
            raise ValueError(f"Bar {start} alone exceeds {MAX_TOKENS} tokens")
        end, tokens, loss_start = accepted
        if any(melody_bars[start : end + 1]) and any(piano_bars[start : end + 1]):
            windows.append({
                "bar_start": start,
                "bar_end": end,
                "tokens": tokens,
                "loss_start": loss_start,
            })
        start = end + 1
    return windows


def initial_cl_tempo(midi: Any) -> int:
    active = 120.0
    for event in sorted(midi.tempo_changes, key=lambda event: int(event.time)):
        if int(event.time) > 0:
            break
        active = float(event.tempo)
    return clamp_tempo(active)


def process_song(
    song_id: str,
    original_root: Path,
    cl_path: Path,
    split: str,
    vocab: dict[str, int],
    index_metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit: dict[str, Any] = {"song_id": song_id, "split": split, "status": "FAIL", "error": ""}
    rows: list[dict[str, Any]] = []
    try:
        original_path = original_root / song_id / f"{song_id}.mid"
        original = miditoolkit.MidiFile(original_path)
        cl_midi = miditoolkit.MidiFile(cl_path)
        tracks = {
            canonical_track_name(instrument.name): instrument
            for instrument in original.instruments
        }
        melody_track = tracks.get("MELODY")
        piano_track = tracks.get("PIANO")
        if melody_track is None or piano_track is None:
            raise ValueError("Original MIDI must contain MELODY and PIANO tracks")
        original_ppq = int(original.ticks_per_beat)
        cl_ppq = int(cl_midi.ticks_per_beat)
        melody, melody_stats = normalize_quantized_notes(melody_track.notes, original_ppq)
        piano, piano_stats = normalize_quantized_notes(piano_track.notes, original_ppq)

        initial_meter = meter_at_zero(cl_midi)
        origin, beat_stats = beat_alignment(
            original_path.with_name("beat_midi.txt"),
            original,
            original_ppq,
            min(melody[0].start, piano[0].start),
            meter_size(initial_meter),
        )
        relative_melody = [
            PianoNote(note.start - origin, note.end - origin, note.pitch, note.velocity)
            for note in melody if note.start >= origin
        ]
        relative_piano = [
            PianoNote(note.start - origin, note.end - origin, note.pitch, note.velocity)
            for note in piano if note.start >= origin
        ]
        if not relative_melody or not relative_piano:
            raise ValueError("No Melody or Piano notes remain after bar alignment")

        key_changes = make_key_changes(cl_midi)
        chord_changes = make_chord_changes(cl_midi)
        maximum_grid = max(
            max(note.start for note in relative_melody),
            max(note.start for note in relative_piano),
            key_changes[-1][0],
            chord_changes[-1][0],
        )
        bars = make_bars(cl_midi, maximum_grid)
        melody_bars = assign_notes_to_bars(relative_melody, bars)
        piano_bars = assign_notes_to_bars(relative_piano, bars)
        tempo = initial_cl_tempo(cl_midi)
        windows = make_windows(
            bars, melody_bars, piano_bars, key_changes, chord_changes, tempo
        )
        if not windows:
            raise ValueError("No non-empty paired windows were produced")

        for window_index, window in enumerate(windows):
            tokens = window["tokens"]
            unknown = sorted({token for token in tokens if token not in vocab})
            if unknown:
                raise ValueError(f"Tokens outside extended vocabulary: {unknown[:12]}")
            rows.append({
                "sample_id": f"pop909-cl::{song_id}::window-{window_index:03d}",
                "song_id": f"pop909::{song_id}",
                "dataset": "POP909+POP909-CL",
                "task": "KEY_CHORD_MELODY_TO_PIANO",
                "split": split,
                "source_bar_start_zero_based": window["bar_start"],
                "source_bar_end_zero_based": window["bar_end"],
                "source_bar_count": window["bar_end"] - window["bar_start"] + 1,
                "token_count": len(tokens),
                "loss_start_token_index": window["loss_start"],
                "tokens": tokens,
                "input_ids": [vocab[token] for token in tokens],
            })
        audit.update({
            "status": "PASS",
            "original_ppq": original_ppq,
            "cl_ppq": cl_ppq,
            "ppq_rescaled": original_ppq != cl_ppq,
            "origin_grid": origin,
            "bar_count": len(bars),
            "window_count": len(rows),
            "key_change_count": len(key_changes),
            "chord_change_count": len(chord_changes),
            "other_chord_count": sum(state.other for _time, state in chord_changes),
            "melody_note_count": len(relative_melody),
            "piano_note_count": len(relative_piano),
            "melody_normalization": melody_stats,
            "piano_normalization": piano_stats,
            "beat_alignment": beat_stats,
        })
    except Exception as exc:
        audit["error"] = f"{type(exc).__name__}: {exc}"
    return audit, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop909-original", type=Path, required=True)
    parser.add_argument("--pop909-cl", type=Path, required=True)
    parser.add_argument("--accepted-songs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-vocab",
        type=Path,
        default=Path(__file__).with_name("remi-plus-vocab-v4.json"),
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path(__file__).with_name("pop909-index.csv"),
    )
    args = parser.parse_args()

    original_root = args.pop909_original.expanduser().resolve()
    cl_root = args.pop909_cl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists; choose a new path: {output_dir}")
    if not original_root.is_dir() or not (cl_root / "POP909_processed").is_dir():
        raise SystemExit("One or both dataset roots are invalid")

    accepted_payload = read_json(args.accepted_songs.expanduser().resolve())
    if not isinstance(accepted_payload, list):
        raise SystemExit("accepted-songs.json must contain a JSON array")
    accepted = [f"{int(value):03d}" for value in accepted_payload]
    if len(accepted) != len(set(accepted)):
        raise SystemExit("accepted-songs.json contains duplicate IDs")
    cl_midis = discover_cl_midis(cl_root)
    index = load_index_metadata(args.index_csv.expanduser().resolve())
    vocab, _id_to_token, vocab_payload = build_extended_vocab(
        args.base_vocab.expanduser().resolve()
    )
    splits = split_map([f"{number:03d}" for number in range(1, 910)])

    output_dir.mkdir(parents=True)
    write_json(output_dir / "remi-plus-vocab-key-chord-v1.json", vocab_payload)
    handles = {
        split: (output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "validation", "test")
    }
    audits: list[dict[str, Any]] = []
    counts = {split: {"songs": 0, "windows": 0} for split in handles}
    first_tokens: list[str] | None = None
    try:
        for position, song_id in enumerate(accepted, 1):
            split = splits[song_id]
            cl_path = cl_midis.get(song_id)
            if cl_path is None:
                audit = {"song_id": song_id, "split": split, "status": "FAIL", "error": "missing CL MIDI"}
                rows: list[dict[str, Any]] = []
            else:
                audit, rows = process_song(
                    song_id, original_root, cl_path, split, vocab, index[song_id]
                )
            audits.append(audit)
            if audit["status"] == "PASS":
                counts[split]["songs"] += 1
                counts[split]["windows"] += len(rows)
                for row in rows:
                    handles[split].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                if first_tokens is None:
                    first_tokens = rows[0]["tokens"]
            if position % 50 == 0 or position == len(accepted):
                print(f"Processed {position}/{len(accepted)}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    failures = [row for row in audits if row["status"] != "PASS"]
    write_json(output_dir / "conversion-audit.json", audits)
    write_json(output_dir / "conversion-failures.json", failures)
    if first_tokens:
        (output_dir / "TOKENIZATION_EXAMPLE.txt").write_text(
            "\n".join(first_tokens) + "\n", encoding="utf-8"
        )
    summary = {
        "format": "POP909 Original MELODY/PIANO + POP909-CL Key/Chord",
        "accepted_song_ids_supplied": len(accepted),
        "songs_passed_conversion": len(accepted) - len(failures),
        "songs_failed_conversion": len(failures),
        "split_counts": counts,
        "base_vocab_size": vocab_payload["base_vocab_size"],
        "extended_vocab_size": vocab_payload["vocab_size"],
        "old_vocab_ids_preserved": True,
        "grid_per_quarter": GRID_PER_QUARTER,
        "max_tokens": MAX_TOKENS,
        "max_bars_per_window": MAX_BARS_PER_WINDOW,
        "bar_format": "BAR_xxx -> Key -> ROLE_CHORD -> ROLE_MELODY -> BAR_END",
        "loss_policy": "labels before target content after ARRANGEMENT_START are masked",
        "ppq_policy": "convert each source tick through its own PPQ to 48-grid",
    }
    write_json(output_dir / "dataset-report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures:
        print(f"WARNING: {len(failures)} songs failed conversion; inspect conversion-failures.json")


if __name__ == "__main__":
    main()
