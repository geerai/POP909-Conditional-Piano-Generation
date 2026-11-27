#!/usr/bin/env python3
"""Build strict POP909 Melody-conditioned Piano REMI+ data.

Only the final ``NNN/NNN.mid`` file from each of the 909 song directories is
eligible. MELODY is serialized before CONDITION_END and PIANO is serialized
after ARRANGEMENT_START. BRIDGE is excluded. The existing 1,727-token REMI+ v4
IDs are kept unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


GRID_PER_QUARTER = 48
MAX_TOKENS = 2048
MAX_DURATION = 768
TEMPO_MIN = 30
TEMPO_MAX = 240
SPLIT_SEED = 20260923
WINDOW_SEED = 20260924
VALIDATION_SONGS = 50
TEST_SONGS = 50
SUPPORTED_INDEX_BEATS_PER_MEASURE = {2, 3}


@dataclass(frozen=True)
class PianoNote:
    start: int
    end: int
    pitch: int
    velocity: int


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_vocab(path: Path) -> tuple[dict[str, int], list[str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(payload["tokens"], key=lambda row: int(row["id"]))
    tokens = [str(row["token"]) for row in rows]
    ids = [int(row["id"]) for row in rows]
    if ids != list(range(len(rows))) or len(tokens) != len(set(tokens)):
        raise ValueError(f"invalid vocabulary: {path}")
    if len(tokens) != 1727:
        raise ValueError(f"expected fixed 1,727-token vocabulary, got {len(tokens)}")
    return dict(zip(tokens, ids)), tokens, payload


def round_half_up_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (2 * int(numerator) + int(denominator)) // (2 * int(denominator))


def tick_to_grid(tick: int, ppq: int) -> int:
    return round_half_up_ratio(int(tick) * GRID_PER_QUARTER, int(ppq))


def grid_to_tick(grid: int, ppq: int) -> int:
    return round_half_up_ratio(int(grid) * int(ppq), GRID_PER_QUARTER)


def clamp_tempo(value: float) -> int:
    return max(TEMPO_MIN, min(TEMPO_MAX, int(math.floor(float(value) + 0.5))))


def grids_per_bar(meter: tuple[int, int]) -> int:
    numerator, denominator = meter
    raw = numerator * 4 * GRID_PER_QUARTER
    if raw % denominator:
        raise ValueError(f"meter {numerator}/{denominator} is not representable")
    return raw // denominator


def canonical_track_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def discover_main_midis(root: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    directories = sorted(
        (path for path in root.iterdir() if path.is_dir() and re.fullmatch(r"\d{3}", path.name)),
        key=lambda path: path.name,
    )
    for directory in directories:
        main = directory / f"{directory.name}.mid"
        if not main.is_file():
            raise FileNotFoundError(f"missing final main MIDI: {main}")
        rows.append((directory.name, main))
    return rows


def split_map(song_ids: Sequence[str]) -> dict[str, str]:
    ordered = sorted(
        song_ids,
        key=lambda song_id: hashlib.sha256(
            f"{SPLIT_SEED}:{song_id}".encode("utf-8")
        ).digest(),
    )
    test = set(ordered[:TEST_SONGS])
    validation = set(ordered[TEST_SONGS:TEST_SONGS + VALIDATION_SONGS])
    return {
        song_id: "test" if song_id in test else "validation" if song_id in validation else "train"
        for song_id in song_ids
    }


def load_index_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Load the checked-in CSV exported from POP909/index.xlsx.

    The legacy MIDI time-signature events are not authoritative: many files
    encode 1/4 even though the official index and beat annotations describe
    two- or three-beat measures.  Keeping the tiny CSV beside this script also
    avoids requiring an Excel reader on the training server.
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "song_id", "name", "artist", "modify_times",
        "num_beats_per_measure", "num_quavers_per_beat",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid POP909 index CSV: {path}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        song_id = f"{int(row['song_id']):03d}"
        if song_id in result:
            raise ValueError(f"duplicate POP909 song_id in index: {song_id}")
        beats = int(row["num_beats_per_measure"])
        quavers = int(row["num_quavers_per_beat"])
        if beats not in SUPPORTED_INDEX_BEATS_PER_MEASURE:
            raise ValueError(f"unsupported official beats per measure for {song_id}: {beats}")
        if quavers not in {2, 3}:
            raise ValueError(f"unsupported official quavers per beat for {song_id}: {quavers}")
        result[song_id] = {
            **row,
            "num_beats_per_measure": beats,
            "num_quavers_per_beat": quavers,
            "meter": (beats, 4),
        }
    return result


def initial_tempo(midi: Any, origin_tick: int) -> int:
    active = 120.0
    for item in sorted(midi.tempo_changes, key=lambda item: int(item.time)):
        if int(item.time) > origin_tick:
            break
        active = float(item.tempo)
    return clamp_tempo(active)


def seconds_to_tick(seconds: float, midi: Any, ppq: int) -> int:
    """Invert the source tempo map for an annotation time in seconds."""
    changes = [
        (int(item.time), float(item.tempo))
        for item in sorted(midi.tempo_changes, key=lambda item: int(item.time))
    ]
    if not changes or changes[0][0] > 0:
        changes.insert(0, (0, 120.0))
    segments: list[tuple[int, float, float]] = []
    elapsed = 0.0
    previous_tick, previous_tempo = changes[0]
    segments.append((previous_tick, elapsed, previous_tempo))
    for tick, tempo in changes[1:]:
        if tick <= previous_tick:
            previous_tempo = tempo
            segments[-1] = (previous_tick, elapsed, previous_tempo)
            continue
        elapsed += (tick - previous_tick) * 60.0 / (previous_tempo * ppq)
        previous_tick = tick
        previous_tempo = tempo
        segments.append((previous_tick, elapsed, previous_tempo))
    active_tick, active_seconds, active_tempo = segments[0]
    for segment in segments:
        if segment[1] > seconds:
            break
        active_tick, active_seconds, active_tempo = segment
    tick = active_tick + (seconds - active_seconds) * active_tempo * ppq / 60.0
    return max(0, int(math.floor(tick + 0.5)))


def beat_alignment(
    path: Path,
    midi: Any,
    ppq: int,
    first_note_grid: int,
    bar_size: int,
) -> tuple[int, dict[str, Any]]:
    """Align bar zero to POP909's first official downbeat candidate.

    In beat_midi.txt, column 1 is time in seconds and column 2 is the primary
    downbeat candidate described by POP909.  Column 3 is retained only as an
    audit count; it is not used to choose barlines.
    """
    if not path.is_file():
        raise FileNotFoundError(f"missing official beat annotation: {path}")
    rows: list[list[float]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            values = [float(value) for value in parts]
        except ValueError as exc:
            raise ValueError(f"invalid beat_midi row {number}: {line!r}") from exc
        if not values:
            continue
        rows.append(values)
    if not rows:
        raise ValueError(f"empty beat annotation: {path}")
    primary_seconds = [row[0] for row in rows if len(row) >= 2 and row[1] > 0.5]
    if not primary_seconds:
        raise ValueError(f"no primary downbeat markers in {path}")
    primary_grids = sorted({
        tick_to_grid(seconds_to_tick(value, midi, ppq), ppq)
        for value in primary_seconds
    })
    first_downbeat = primary_grids[0]
    origin_grid = first_downbeat
    while origin_grid > first_note_grid:
        origin_grid -= bar_size
    while origin_grid + bar_size <= first_note_grid:
        origin_grid += bar_size
    intervals = [right - left for left, right in zip(primary_grids, primary_grids[1:])]
    median_interval = ""
    median_error = ""
    if intervals:
        ordered = sorted(intervals)
        median_interval = ordered[len(ordered) // 2]
        median_error = abs(int(median_interval) - bar_size)
    return origin_grid, {
        "beat_midi_present": True,
        "beat_row_count": len(rows),
        "primary_downbeat_marker_count": len(primary_seconds),
        "secondary_downbeat_marker_count": sum(
            bool(len(row) >= 3 and row[2] > 0.5) for row in rows
        ),
        "first_beat_seconds": rows[0][0] if rows else "",
        "first_primary_downbeat_seconds": primary_seconds[0],
        "first_primary_downbeat_grid": first_downbeat,
        "median_primary_downbeat_interval_grid": median_interval,
        "median_primary_downbeat_interval_error_grid": median_error,
        "bar_alignment_source": "POP909 beat_midi.txt column 2",
    }


def normalize_quantized_notes(
    notes: Sequence[Any], ppq: int,
) -> tuple[list[PianoNote], dict[str, int]]:
    raw_valid = [
        note for note in notes
        if int(note.end) > int(note.start) and 0 <= int(note.pitch) <= 127
    ]
    exact: dict[tuple[int, int, int], PianoNote] = {}
    for note in raw_valid:
        start = tick_to_grid(int(note.start), ppq)
        end = max(start + 1, tick_to_grid(int(note.end), ppq))
        key = (start, end, int(note.pitch))
        candidate = PianoNote(start, end, int(note.pitch), max(1, min(127, int(note.velocity))))
        if key not in exact or candidate.velocity > exact[key].velocity:
            exact[key] = candidate
    exact_duplicates = len(raw_valid) - len(exact)

    by_onset_pitch: dict[tuple[int, int], PianoNote] = {}
    same_onset_collisions = 0
    for note in exact.values():
        key = (note.start, note.pitch)
        previous = by_onset_pitch.get(key)
        if previous is None:
            by_onset_pitch[key] = note
        else:
            same_onset_collisions += 1
            by_onset_pitch[key] = max(
                (previous, note),
                key=lambda item: (item.end, item.velocity),
            )

    by_pitch: dict[int, list[PianoNote]] = {}
    for note in by_onset_pitch.values():
        by_pitch.setdefault(note.pitch, []).append(note)
    resolved: list[PianoNote] = []
    retrigger_clips = 0
    for pitch, group in by_pitch.items():
        ordered = sorted(group, key=lambda item: (item.start, item.end, -item.velocity))
        output: list[PianoNote] = []
        for note in ordered:
            if output and note.start < output[-1].end:
                previous = output[-1]
                output[-1] = PianoNote(
                    previous.start,
                    note.start,
                    previous.pitch,
                    previous.velocity,
                )
                retrigger_clips += 1
            output.append(note)
        resolved.extend(item for item in output if item.end > item.start)
    resolved.sort(key=lambda item: (item.start, item.pitch, item.end, item.velocity))
    if not resolved:
        raise ValueError("PIANO track contains no usable notes")
    return resolved, {
        "raw_piano_note_count": len(notes),
        "valid_piano_note_count": len(raw_valid),
        "exact_duplicate_count": exact_duplicates,
        "same_onset_pitch_collision_count": same_onset_collisions,
        "same_pitch_retrigger_clip_count": retrigger_clips,
        "normalized_note_count": len(resolved),
    }


def serialize_window(
    melody_bars: Sequence[list[tuple[int, int, int, int]]],
    piano_bars: Sequence[list[tuple[int, int, int, int]]],
    meter: tuple[int, int],
    tempo: int,
    start: int,
    end: int,
) -> tuple[list[str], int]:
    tokens: list[str] = []
    if start == 0:
        tokens.append("<SONG_START>")
    tokens.extend([
        "<TASK_MELODY_TO_ROLE>",
        f"<METER_{meter[0]}_{meter[1]}>",
        f"<TEMPO_{tempo:03d}>",
        "<ROLE_MELODY>",
        "<MELODY_PIANO_000>",
        "<ROLE_PIANO>",
        "<PIANO_000>",
        "<CONDITION_START>",
    ])
    for bar_index in range(start, end + 1):
        tokens.extend(["<BAR>", "<ROLE_MELODY>"])
        for position, pitch, duration, _velocity in melody_bars[bar_index]:
            tokens.extend([
                f"<POSITION_{position:03d}>",
                f"<PITCH_{pitch:03d}>",
                f"<DURATION_{duration:03d}>",
            ])
        tokens.extend(["<ROLE_END>", "<BAR_END>"])
    tokens.extend([
        "<CONDITION_END>",
        "<ARRANGEMENT_START>",
    ])
    loss_start = len(tokens)
    for bar_index in range(start, end + 1):
        tokens.extend(["<BAR>", "<ROLE_PIANO>"])
        for position, pitch, duration, _velocity in piano_bars[bar_index]:
            tokens.extend([
                f"<POSITION_{position:03d}>",
                f"<PITCH_{pitch:03d}>",
                f"<DURATION_{duration:03d}>",
            ])
        tokens.extend(["<ROLE_END>", "<BAR_END>"])
    if end == len(piano_bars) - 1:
        tokens.extend(["<ARRANGEMENT_END>", "<SONG_END>"])
    return tokens, loss_start


def make_windows(
    melody_bars: Sequence[list[tuple[int, int, int, int]]],
    piano_bars: Sequence[list[tuple[int, int, int, int]]],
    meter: tuple[int, int],
    tempo: int,
    song_id: str,
) -> list[dict[str, Any]]:
    rng = random.Random(f"{WINDOW_SEED}:{song_id}")
    result: list[dict[str, Any]] = []
    start, previous_end = 0, -1
    if len(melody_bars) != len(piano_bars):
        raise ValueError("condition and target must share the same bar count")
    while start < len(piano_bars):
        accepted: tuple[int, list[str], int] | None = None
        for end in range(start, len(piano_bars)):
            tokens, loss_start = serialize_window(
                melody_bars, piano_bars, meter, tempo, start, end
            )
            if len(tokens) > MAX_TOKENS:
                break
            accepted = end, tokens, loss_start
        if accepted is None:
            raise ValueError(f"bar {start + 1} alone exceeds {MAX_TOKENS} tokens")
        end, tokens, loss_start = accepted
        result.append({
            "bar_start": start,
            "bar_end": end,
            "overlap": max(0, previous_end - start + 1),
            "tokens": tokens,
            "loss_start": loss_start,
        })
        if end == len(piano_bars) - 1:
            break
        span = end - start + 1
        overlap_target = rng.randint(
            max(1, math.ceil(span * 0.25)),
            max(1, math.ceil(span * 0.50)),
        )
        previous_end = end
        start = max(start + 1, end - overlap_target + 1)
    return result


def write_cleaned_midi(
    path: Path,
    ppq: int,
    meter: tuple[int, int],
    tempo: int,
    relative_notes: Sequence[PianoNote],
) -> None:
    import miditoolkit  # type: ignore

    midi = miditoolkit.MidiFile(ticks_per_beat=ppq)
    midi.tempo_changes = [miditoolkit.TempoChange(tempo, 0)]
    midi.time_signature_changes = [
        miditoolkit.TimeSignature(meter[0], meter[1], 0)
    ]
    instrument = miditoolkit.Instrument(program=0, is_drum=False, name="PIANO")
    instrument.notes = [
        miditoolkit.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=grid_to_tick(note.start, ppq),
            end=grid_to_tick(note.end, ppq),
        )
        for note in relative_notes
    ]
    midi.instruments = [instrument]
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(path))


def write_roundtrip_midi(
    path: Path,
    tokens: Sequence[str],
    ppq: int = 480,
) -> tuple[int, int]:
    import miditoolkit  # type: ignore

    meter = (4, 4)
    tempo = 120
    for token in tokens:
        match = re.fullmatch(r"<METER_(\d+)_(\d+)>", token)
        if match:
            meter = (int(match.group(1)), int(match.group(2)))
        match = re.fullmatch(r"<TEMPO_(\d+)>", token)
        if match:
            tempo = int(match.group(1))
    bar_size = grids_per_bar(meter)
    bar_index = -1
    in_target = False
    notes: list[tuple[int, int, int]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "<ARRANGEMENT_START>":
            in_target = True
            bar_index = -1
            index += 1
            continue
        if token == "<BAR>":
            if in_target:
                bar_index += 1
            index += 1
            continue
        position = re.fullmatch(r"<POSITION_(\d+)>", token)
        if position and in_target:
            if index + 2 >= len(tokens):
                raise ValueError("truncated note triple")
            pitch = re.fullmatch(r"<PITCH_(\d+)>", tokens[index + 1])
            duration = re.fullmatch(r"<DURATION_(\d+)>", tokens[index + 2])
            if not pitch or not duration or bar_index < 0:
                raise ValueError(f"invalid note triple at token {index}")
            start = bar_index * bar_size + int(position.group(1))
            notes.append((start, int(pitch.group(1)), int(duration.group(1))))
            index += 3
            continue
        index += 1
    midi = miditoolkit.MidiFile(ticks_per_beat=ppq)
    midi.tempo_changes = [miditoolkit.TempoChange(tempo, 0)]
    midi.time_signature_changes = [
        miditoolkit.TimeSignature(meter[0], meter[1], 0)
    ]
    instrument = miditoolkit.Instrument(program=0, is_drum=False, name="PIANO")
    instrument.notes = [
        miditoolkit.Note(
            velocity=80,
            pitch=pitch,
            start=grid_to_tick(start, ppq),
            end=grid_to_tick(start + duration, ppq),
        )
        for start, pitch, duration in notes
    ]
    midi.instruments = [instrument]
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(path))
    return len(notes), bar_index + 1


def process_song(
    song_id: str,
    source: Path,
    split: str,
    output_dir: Path,
    vocab: dict[str, int],
    index_metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    import miditoolkit  # type: ignore

    audit: dict[str, Any] = {
        "song_id": song_id,
        "split": split,
        "source_relative_path": f"{song_id}/{song_id}.mid",
        "status": "FAIL",
        "error": "",
    }
    manifest: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    try:
        midi = miditoolkit.MidiFile(str(source))
        ppq = int(midi.ticks_per_beat)
        if ppq <= 0:
            raise ValueError("invalid PPQ")
        names = [str(instrument.name or "") for instrument in midi.instruments]
        piano_tracks = [
            instrument for instrument in midi.instruments
            if canonical_track_name(str(instrument.name or "")) == "PIANO"
        ]
        melody_tracks = [
            instrument for instrument in midi.instruments
            if canonical_track_name(str(instrument.name or "")) == "MELODY"
        ]
        bridge_tracks = [
            instrument for instrument in midi.instruments
            if canonical_track_name(str(instrument.name or "")) == "BRIDGE"
        ]
        if len(piano_tracks) != 1:
            raise ValueError(f"expected exactly one PIANO track, found {len(piano_tracks)}: {names}")
        if len(melody_tracks) != 1:
            raise ValueError(f"expected exactly one MELODY track, found {len(melody_tracks)}: {names}")
        piano = piano_tracks[0]
        melody = melody_tracks[0]
        if piano.is_drum:
            raise ValueError("PIANO track is marked as drums")
        if melody.is_drum:
            raise ValueError("MELODY track is marked as drums")
        meter = tuple(index_metadata["meter"])
        bar_size = grids_per_bar(meter)
        normalized_piano, piano_note_audit = normalize_quantized_notes(piano.notes, ppq)
        normalized_melody, melody_note_audit = normalize_quantized_notes(melody.notes, ppq)
        first_grid = min(
            min(note.start for note in normalized_piano),
            min(note.start for note in normalized_melody),
        )
        origin_grid, beats = beat_alignment(
            source.with_name("beat_midi.txt"),
            midi,
            ppq,
            first_grid,
            bar_size,
        )
        origin_tick = max(0, grid_to_tick(origin_grid, ppq))
        unclipped_relative_piano = [
            PianoNote(
                note.start - origin_grid,
                note.end - origin_grid,
                note.pitch,
                note.velocity,
            )
            for note in normalized_piano
        ]
        unclipped_relative_melody = [
            PianoNote(
                note.start - origin_grid,
                note.end - origin_grid,
                note.pitch,
                note.velocity,
            )
            for note in normalized_melody
        ]
        clipped_durations = 0

        def clip_notes(notes: Sequence[PianoNote]) -> list[PianoNote]:
            nonlocal clipped_durations
            result: list[PianoNote] = []
            for note in notes:
                # Notes before the official bar origin cannot be represented.
                if note.start < 0:
                    continue
                duration = note.end - note.start
                if duration > MAX_DURATION:
                    duration = MAX_DURATION
                    clipped_durations += 1
                result.append(PianoNote(
                    note.start, note.start + duration, note.pitch, note.velocity
                ))
            return result

        relative_piano = clip_notes(unclipped_relative_piano)
        relative_melody = clip_notes(unclipped_relative_melody)
        if not relative_piano or not relative_melody:
            raise ValueError("MELODY and PIANO must both contain notes after alignment")
        tempo = initial_tempo(midi, origin_tick)
        last_onset = max(
            max(note.start for note in relative_piano),
            max(note.start for note in relative_melody),
        )
        bar_count = last_onset // bar_size + 1

        def notes_to_bars(notes: Sequence[PianoNote]) -> list[list[tuple[int, int, int, int]]]:
            bars: list[list[tuple[int, int, int, int]]] = [[] for _ in range(bar_count)]
            for note in notes:
                bar_index, position = divmod(note.start, bar_size)
                duration = note.end - note.start
                bars[bar_index].append((position, note.pitch, duration, note.velocity))
            for bar in bars:
                bar.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            return bars

        melody_bars = notes_to_bars(relative_melody)
        piano_bars = notes_to_bars(relative_piano)
        candidate_windows = make_windows(melody_bars, piano_bars, meter, tempo, song_id)
        windows = [
            window for window in candidate_windows
            if any(melody_bars[index] for index in range(window["bar_start"], window["bar_end"] + 1))
            and any(piano_bars[index] for index in range(window["bar_start"], window["bar_end"] + 1))
        ]
        empty_role_windows_removed = len(candidate_windows) - len(windows)
        if not windows:
            raise ValueError("all paired windows have an empty MELODY or PIANO side")
        for window_index, window in enumerate(windows):
            tokens = window["tokens"]
            unknown = sorted({token for token in tokens if token not in vocab})
            if unknown:
                raise ValueError(f"tokens outside vocabulary: {unknown[:8]}")
            row = {
                "sample_id": f"pop909::{song_id}::window-{window_index:03d}",
                "song_id": f"pop909::{song_id}",
                "dataset": "POP909",
                "task": "MELODY_TO_PIANO_ACCOMPANIMENT",
                "condition_role": "MELODY",
                "target_roles": ["PIANO"],
                "split": split,
                "source_relative_path": f"{song_id}/{song_id}.mid",
                "condition_track_name": str(melody.name),
                "target_track_name": str(piano.name),
                "source_bar_start_one_based": window["bar_start"] + 1,
                "source_bar_end_one_based": window["bar_end"] + 1,
                "source_bar_count": window["bar_end"] - window["bar_start"] + 1,
                "overlap_with_previous_bars": window["overlap"],
                "contains_true_song_start": window["bar_start"] == 0,
                "contains_true_song_end": window["bar_end"] == len(piano_bars) - 1,
                "token_count": len(tokens),
                "loss_start_token_index": window["loss_start"],
                "input_ids": [vocab[token] for token in tokens],
            }
            rows.append(row)
            manifest.append({key: value for key, value in row.items() if key != "input_ids"})

        cleaned_relative = Path("cleaned-midi") / split / f"{song_id}.mid"
        write_cleaned_midi(
            output_dir / cleaned_relative,
            ppq,
            meter,
            tempo,
            relative_piano,
        )
        audit.update({
            "status": "PASS",
            "track_names": " | ".join(names),
            "piano_track_count": len(piano_tracks),
            "melody_track_count": len(melody_tracks),
            "bridge_track_count_ignored": len(bridge_tracks),
            "melody_note_count": len(melody.notes),
            "bridge_note_count_ignored": sum(len(item.notes) for item in bridge_tracks),
            "piano_program": int(piano.program),
            "piano_control_change_count_audit_only": len(piano.control_changes),
            "piano_pitch_bend_count_audit_only": len(piano.pitch_bends),
            "ppq": ppq,
            "meter": f"{meter[0]}/{meter[1]}",
            "meter_source": "POP909 index.xlsx num_beats_per_measure / 4",
            "official_num_beats_per_measure": index_metadata["num_beats_per_measure"],
            "official_num_quavers_per_beat": index_metadata["num_quavers_per_beat"],
            "official_song_name": index_metadata["name"],
            "official_artist": index_metadata["artist"],
            "tempo": tempo,
            "original_tempo_change_count": len(midi.tempo_changes),
            "original_meter_change_count": len(midi.time_signature_changes),
            "origin_grid_removed": origin_grid,
            "bar_count": bar_count,
            "window_count": len(rows),
            "candidate_window_count": len(candidate_windows),
            "empty_role_windows_removed": empty_role_windows_removed,
            "duration_clipped_to_768_count": clipped_durations,
            "cleaned_midi_relative_path": cleaned_relative.as_posix(),
            **{f"piano_{key}": value for key, value in piano_note_audit.items()},
            **{f"melody_{key}": value for key, value in melody_note_audit.items()},
            **beats,
        })
    except Exception as exc:
        audit["error"] = f"{type(exc).__name__}: {exc}"
    return audit, rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path(__file__).with_name("remi-plus-vocab-v4.json"),
    )
    parser.add_argument("--expected-song-count", type=int, default=909)
    parser.add_argument("--roundtrip-examples", type=int, default=20)
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path(__file__).with_name("pop909-index.csv"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"source directory does not exist: {source_root}")
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite to replace: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    discovered = discover_main_midis(source_root)
    if len(discovered) != args.expected_song_count:
        raise SystemExit(
            f"expected {args.expected_song_count} final main MIDIs, found {len(discovered)}"
        )
    all_midis = list(source_root.rglob("*.mid")) + list(source_root.rglob("*.midi"))
    ignored_version_count = len(all_midis) - len(discovered)
    vocab, id_to_token, vocab_payload = load_vocab(args.vocab)
    index_rows = load_index_metadata(args.index_csv)
    discovered_ids = {song_id for song_id, _path in discovered}
    if set(index_rows) != discovered_ids:
        raise SystemExit(
            "POP909 index/song mismatch: "
            f"index_only={sorted(set(index_rows) - discovered_ids)[:8]} "
            f"midi_only={sorted(discovered_ids - set(index_rows))[:8]}"
        )
    write_json(output_dir / "remi-plus-vocab-v4.json", vocab_payload)
    splits = split_map([song_id for song_id, _path in discovered])
    write_csv(
        output_dir / "song-split.csv",
        ({"song_id": song_id, "split": splits[song_id]} for song_id, _ in discovered),
        ["song_id", "split"],
    )

    handles = {
        split: (output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "validation", "test")
    }
    audits: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    split_counts = {
        split: {"songs_assigned": 0, "songs_passed": 0, "songs_failed": 0, "windows": 0}
        for split in handles
    }
    first_train_tokens: list[str] | None = None
    roundtrip_candidates: list[tuple[str, str, list[str]]] = []
    try:
        for index, (song_id, source) in enumerate(discovered, 1):
            split = splits[song_id]
            split_counts[split]["songs_assigned"] += 1
            audit, rows, window_rows = process_song(
                song_id, source, split, output_dir, vocab, index_rows[song_id]
            )
            audits.append(audit)
            if audit["status"] == "PASS":
                split_counts[split]["songs_passed"] += 1
                split_counts[split]["windows"] += len(rows)
                for row in rows:
                    handles[split].write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                manifest.extend(window_rows)
                if split == "train" and first_train_tokens is None:
                    first_train_tokens = [id_to_token[value] for value in rows[0]["input_ids"]]
                if len(roundtrip_candidates) < args.roundtrip_examples:
                    roundtrip_candidates.append((split, song_id, [id_to_token[value] for value in rows[0]["input_ids"]]))
            else:
                split_counts[split]["songs_failed"] += 1
            if index % 50 == 0 or index == len(discovered):
                print(f"processed {index}/{len(discovered)} {split_counts}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    audit_fields = [
        "song_id", "split", "source_relative_path", "status", "error", "track_names",
        "piano_track_count", "melody_track_count", "bridge_track_count_ignored",
        "melody_note_count", "bridge_note_count_ignored", "piano_program",
        "piano_raw_piano_note_count", "piano_valid_piano_note_count",
        "piano_exact_duplicate_count", "piano_same_onset_pitch_collision_count",
        "piano_same_pitch_retrigger_clip_count", "piano_normalized_note_count",
        "melody_raw_piano_note_count", "melody_valid_piano_note_count",
        "melody_exact_duplicate_count", "melody_same_onset_pitch_collision_count",
        "melody_same_pitch_retrigger_clip_count", "melody_normalized_note_count",
        "duration_clipped_to_768_count",
        "piano_control_change_count_audit_only", "piano_pitch_bend_count_audit_only",
        "ppq", "meter", "meter_source", "official_num_beats_per_measure",
        "official_num_quavers_per_beat", "official_song_name", "official_artist",
        "tempo", "original_tempo_change_count",
        "original_meter_change_count", "origin_grid_removed", "bar_count", "window_count",
        "candidate_window_count", "empty_role_windows_removed",
        "beat_midi_present", "beat_row_count", "primary_downbeat_marker_count",
        "secondary_downbeat_marker_count", "first_beat_seconds",
        "first_primary_downbeat_seconds", "first_primary_downbeat_grid",
        "median_primary_downbeat_interval_grid",
        "median_primary_downbeat_interval_error_grid", "bar_alignment_source",
        "cleaned_midi_relative_path",
    ]
    write_csv(output_dir / "conversion-audit.csv", audits, audit_fields)
    manifest_fields = [
        "sample_id", "song_id", "dataset", "task", "condition_role", "target_roles", "split",
        "source_relative_path", "condition_track_name", "target_track_name", "source_bar_start_one_based",
        "source_bar_end_one_based", "source_bar_count", "overlap_with_previous_bars",
        "contains_true_song_start", "contains_true_song_end", "token_count",
        "loss_start_token_index",
    ]
    write_csv(output_dir / "window-manifest.csv", manifest, manifest_fields)
    if first_train_tokens is None:
        raise SystemExit("no valid Train samples were produced")
    (output_dir / "TOKENIZATION_EXAMPLE.txt").write_text(
        "\n".join(first_train_tokens) + "\n", encoding="utf-8"
    )

    roundtrip_rows: list[dict[str, Any]] = []
    for split, song_id, tokens in roundtrip_candidates:
        target = output_dir / "roundtrip-examples" / split / f"{song_id}__window-000.mid"
        note_count, bar_count = write_roundtrip_midi(target, tokens)
        roundtrip_rows.append({
            "song_id": song_id,
            "split": split,
            "status": "PASS",
            "decoded_note_count": note_count,
            "decoded_bar_count": bar_count,
            "midi_relative_path": target.relative_to(output_dir).as_posix(),
        })
    write_csv(
        output_dir / "roundtrip-audit.csv",
        roundtrip_rows,
        ["song_id", "split", "status", "decoded_note_count", "decoded_bar_count", "midi_relative_path"],
    )

    summary = {
        "format": "POP909 final main MIDI -> MELODY condition + PIANO target -> REMI+ v4",
        "source_root": str(source_root),
        "main_midi_count": len(discovered),
        "ignored_versions_or_other_midi_count": ignored_version_count,
        "track_policy": "use exactly one MELODY as condition and one PIANO as target; ignore BRIDGE",
        "meter_policy": "use POP909 index.xlsx beats-per-measure as N/4; align bars to beat_midi.txt primary downbeats; ignore unreliable legacy MIDI time-signature events",
        "task_token_emitted": True,
        "task_token": "<TASK_MELODY_TO_ROLE>",
        "vocabulary_size": len(vocab),
        "vocabulary_ids_changed": False,
        "grid_per_quarter": GRID_PER_QUARTER,
        "maximum_tokens": MAX_TOKENS,
        "maximum_duration_grid": MAX_DURATION,
        "velocity_in_tokens": False,
        "controller_events_in_tokens": False,
        "split_policy": "fixed SHA-256 song split: 809 Train / 50 Validation / 50 Test before conversion",
        "split_seed": SPLIT_SEED,
        "window_seed": WINDOW_SEED,
        "window_policy": "paired condition/target complete bars; maximum fitting window; deterministic-random 25%-50% bar overlap",
        "loss_policy": "mask through ARRANGEMENT_START; optimize only PIANO target tokens",
        "pitch_augmentation": "none",
        "split_counts": split_counts,
        "roundtrip_example_count": len(roundtrip_rows),
        "test_opened_during_training": False,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
