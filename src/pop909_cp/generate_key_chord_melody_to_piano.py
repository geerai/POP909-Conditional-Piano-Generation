#!/usr/bin/env python3
"""Generate POP909 Piano accompaniments with REMI-style top-k sampling."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import Qwen3ForCausalLM, StoppingCriteria, StoppingCriteriaList, set_seed

try:
    from .validate_key_chord_melody_to_piano import validate_tokens
except ImportError:  # Support direct execution from this directory.
    from validate_key_chord_melody_to_piano import validate_tokens

try:
    import miditoolkit
except ImportError as exc:
    raise SystemExit("Install miditoolkit in the qwen-midi environment.") from exc


GRID_PER_QUARTER = 48
PPQ = 480
METHOD = "remi_temperature_topk_sampling"
PC_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
PC_FROM_NAME = {name.replace("#", "_SHARP"): index for index, name in enumerate(PC_NAMES)}
CHORD_INTERVALS = {
    "M": (0, 4, 7), "m": (0, 3, 7), "o": (0, 3, 6), "aug": (0, 4, 8),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "D7": (0, 4, 7, 10),
    "M7": (0, 4, 7, 11), "m7": (0, 3, 7, 10), "half_o7": (0, 3, 6, 10),
    "o7": (0, 3, 6, 9), "mM7": (0, 3, 7, 11), "aug7": (0, 4, 8, 10),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_vocab(path: Path) -> tuple[list[str], dict[str, int]]:
    rows = sorted(read_json(path)["tokens"], key=lambda row: int(row["id"]))
    tokens = [str(row["token"]) for row in rows]
    if [int(row["id"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Vocabulary IDs are not contiguous")
    return tokens, {token: index for index, token in enumerate(tokens)}


def parse_meter_tempo(tokens: Sequence[str]) -> tuple[tuple[int, int], int]:
    meter = None
    tempo = None
    for token in tokens:
        match = re.fullmatch(r"<METER_(\d+)_(\d+)>", token)
        if match:
            meter = (int(match.group(1)), int(match.group(2)))
        match = re.fullmatch(r"<TEMPO_(\d+)>", token)
        if match:
            tempo = int(match.group(1))
    if meter is None or tempo is None:
        raise ValueError("Prompt lacks meter or tempo")
    return meter, tempo


def bar_size(meter: tuple[int, int]) -> int:
    return meter[0] * 4 * GRID_PER_QUARTER // meter[1]


def condition_bar_count(tokens: Sequence[str]) -> int:
    start = tokens.index("<CONDITION_START>")
    end = tokens.index("<CONDITION_END>")
    bars = [token for token in tokens[start + 1:end] if re.fullmatch(r"<BAR_\d{3}>", token)]
    if not bars:
        raise ValueError("Condition contains no bars")
    expected = [f"<BAR_{index:03d}>" for index in range(len(bars))]
    if bars != expected:
        raise ValueError("Condition bar IDs are not contiguous")
    return len(bars)


class StopAfterBars(StoppingCriteria):
    def __init__(self, prompt_length: int, bar_end_id: int, arrangement_end_id: int,
                 song_end_id: int, expected_bars: int) -> None:
        self.prompt_length = prompt_length
        self.bar_end_id = bar_end_id
        self.arrangement_end_id = arrangement_end_id
        self.song_end_id = song_end_id
        self.expected_bars = expected_bars

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        generated = input_ids[0, self.prompt_length:]
        if generated.numel() == 0:
            return False
        last = int(generated[-1])
        if last in {self.arrangement_end_id, self.song_end_id}:
            return True
        return int((generated == self.bar_end_id).sum()) >= self.expected_bars


def parse_note_target(tokens: Sequence[str], meter: tuple[int, int]) -> tuple[list[tuple[int, int, int]], list[str], int]:
    size = bar_size(meter)
    notes: list[tuple[int, int, int]] = []
    errors: list[str] = []
    current_bar: int | None = None
    bars_seen: set[int] = set()
    index = 0
    while index < len(tokens):
        bar_match = re.fullmatch(r"<BAR_(\d{3})>", tokens[index])
        if bar_match:
            current_bar = int(bar_match.group(1))
            bars_seen.add(current_bar)
            index += 1
            continue
        position_match = re.fullmatch(r"<POSITION_(\d{3})>", tokens[index])
        if position_match:
            triple = tokens[index:index + 3]
            if current_bar is None or len(triple) != 3:
                errors.append(f"orphan/truncated position at token {index}")
                index += 1
                continue
            pitch_match = re.fullmatch(r"<PITCH_(\d{3})>", triple[1])
            duration_match = re.fullmatch(r"<DURATION_(\d{3})>", triple[2])
            position = int(position_match.group(1))
            if not pitch_match or not duration_match or position >= size:
                errors.append(f"invalid note triple at token {index}: {triple}")
                index += 1
                continue
            pitch = int(pitch_match.group(1))
            duration = int(duration_match.group(1))
            if not 0 <= pitch <= 127 or not 1 <= duration <= 768:
                errors.append(f"out-of-range note at token {index}: {triple}")
            else:
                notes.append((current_bar * size + position, pitch, duration))
            index += 3
            continue
        index += 1
    return notes, errors, len(bars_seen)


def parse_condition_melody(tokens: Sequence[str], meter: tuple[int, int]) -> list[tuple[int, int, int]]:
    start = tokens.index("<CONDITION_START>") + 1
    end = tokens.index("<CONDITION_END>")
    body = tokens[start:end]
    size = bar_size(meter)
    notes: list[tuple[int, int, int]] = []
    current_bar = 0
    in_melody = False
    index = 0
    while index < len(body):
        match = re.fullmatch(r"<BAR_(\d{3})>", body[index])
        if match:
            current_bar = int(match.group(1))
        elif body[index] == "<ROLE_MELODY>":
            in_melody = True
        elif body[index] == "<ROLE_END>" and in_melody:
            in_melody = False
        elif in_melody:
            position = re.fullmatch(r"<POSITION_(\d{3})>", body[index])
            if position and index + 2 < len(body):
                pitch = re.fullmatch(r"<PITCH_(\d{3})>", body[index + 1])
                duration = re.fullmatch(r"<DURATION_(\d{3})>", body[index + 2])
                if pitch and duration:
                    notes.append((current_bar * size + int(position.group(1)), int(pitch.group(1)), int(duration.group(1))))
                    index += 2
        index += 1
    return notes


def parse_condition_chords(tokens: Sequence[str], meter: tuple[int, int]) -> list[tuple[int, int, tuple[int, ...]]]:
    start = tokens.index("<CONDITION_START>") + 1
    end = tokens.index("<CONDITION_END>")
    body = tokens[start:end]
    size = bar_size(meter)
    changes: list[tuple[int, tuple[int, ...]]] = []
    current_bar = 0
    in_chord = False
    index = 0
    while index < len(body):
        match = re.fullmatch(r"<BAR_(\d{3})>", body[index])
        if match:
            current_bar = int(match.group(1))
        elif body[index] == "<ROLE_CHORD>":
            in_chord = True
        elif body[index] == "<ROLE_END>" and in_chord:
            in_chord = False
        elif in_chord:
            position = re.fullmatch(r"<POSITION_(\d{3})>", body[index])
            if position and index + 1 < len(body):
                time = current_bar * size + int(position.group(1))
                next_token = body[index + 1]
                if next_token in {"<NO_CHORD>", "<CHORD_OTHER>"}:
                    changes.append((time, ()))
                    index += 1
                elif index + 3 < len(body):
                    root = re.fullmatch(r"<CHORD_ROOT_([A-Z_]+)>", body[index + 1])
                    quality = re.fullmatch(r"<CHORD_QUALITY_([A-Za-z0-9_]+)>", body[index + 2])
                    if root and quality and root.group(1) in PC_FROM_NAME and quality.group(1) in CHORD_INTERVALS:
                        root_pc = PC_FROM_NAME[root.group(1)]
                        pitches = tuple(48 + root_pc + interval for interval in CHORD_INTERVALS[quality.group(1)])
                        changes.append((time, pitches))
                        index += 3
        index += 1
    result: list[tuple[int, int, tuple[int, ...]]] = []
    total_end = condition_bar_count(tokens) * size
    for change_index, (time, pitches) in enumerate(changes):
        end_time = changes[change_index + 1][0] if change_index + 1 < len(changes) else total_end
        if pitches and end_time > time:
            result.append((time, end_time, pitches))
    return result


def write_midi(path: Path, meter: tuple[int, int], tempo: int,
               tracks: Sequence[tuple[str, Sequence[tuple[int, int, int]], int]]) -> None:
    midi = miditoolkit.MidiFile(ticks_per_beat=PPQ)
    midi.tempo_changes = [miditoolkit.TempoChange(tempo, 0)]
    midi.time_signature_changes = [miditoolkit.TimeSignature(meter[0], meter[1], 0)]
    instruments = []
    for name, notes, velocity in tracks:
        instrument = miditoolkit.Instrument(program=0, is_drum=False, name=name)
        instrument.notes = [
            miditoolkit.Note(
                velocity=velocity,
                pitch=pitch,
                start=round(start * PPQ / GRID_PER_QUARTER),
                end=round((start + duration) * PPQ / GRID_PER_QUARTER),
            )
            for start, pitch, duration in notes if duration > 0
        ]
        instruments.append(instrument)
    midi.instruments = instruments
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(path))


def chord_notes(events: Sequence[tuple[int, int, tuple[int, ...]]]) -> list[tuple[int, int, int]]:
    return [(start, pitch, end - start) for start, end, pitches in events for pitch in pitches]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--selection-seed", type=int, default=20261016)
    parser.add_argument("--generation-seed", type=int, default=20261016)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"Output already exists; choose a new directory: {output_dir}")
    vocab, token_to_id = load_vocab(data_dir / "remi-plus-vocab-key-chord-v1.json")
    model_vocab, _ = load_vocab(model_path / "remi-plus-vocab-key-chord-v1.json")
    if vocab != model_vocab:
        raise ValueError("Model and dataset vocabularies differ")
    rows = [json.loads(line) for line in (data_dir / "test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 1 <= args.num_samples <= len(rows):
        raise ValueError("num-samples is outside the Test-set size")
    selected = random.Random(args.selection_seed).sample(rows, args.num_samples)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    model = Qwen3ForCausalLM.from_pretrained(str(model_path), dtype=dtype).to(device)
    model.eval()
    model.config.use_cache = True
    context_length = int(model.config.max_position_embeddings)
    output_dir.mkdir(parents=True)
    summary_rows: list[dict[str, Any]] = []

    for sample_index, row in enumerate(selected, 1):
        seed = args.generation_seed + sample_index - 1
        set_seed(seed)
        all_ids = [int(value) for value in row["input_ids"]]
        loss_start = int(row["loss_start_token_index"])
        prompt_ids = all_ids[:loss_start]
        reference_ids = all_ids[loss_start:]
        prompt_tokens = [vocab[value] for value in prompt_ids]
        reference_tokens = [vocab[value] for value in reference_ids]
        expected_bars = condition_bar_count(prompt_tokens)
        meter, tempo = parse_meter_tempo(prompt_tokens)
        available = context_length - len(prompt_ids)
        if available <= 0:
            raise ValueError(f"Prompt exceeds context for {row['sample_id']}")
        stopping = StoppingCriteriaList([StopAfterBars(
            len(prompt_ids), token_to_id["<BAR_END>"], token_to_id["<ARRANGEMENT_END>"],
            token_to_id["<SONG_END>"], expected_bars,
        )])
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=inputs,
                max_new_tokens=available,
                do_sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=1.0,
                repetition_penalty=1.0,
                stopping_criteria=stopping,
                pad_token_id=token_to_id["<PAD>"],
                use_cache=True,
            )
        generated_ids = generated[0, len(prompt_ids):].tolist()
        generated_tokens = [vocab[value] for value in generated_ids]
        generated_notes, grammar_errors, generated_bars = parse_note_target(generated_tokens, meter)
        reference_notes, reference_errors, reference_bars = parse_note_target(reference_tokens, meter)
        try:
            validate_tokens(prompt_tokens + generated_tokens, len(prompt_tokens))
        except ValueError as exc:
            grammar_errors.append(str(exc))
        try:
            validate_tokens(prompt_tokens + reference_tokens, len(prompt_tokens))
        except ValueError as exc:
            reference_errors.append(str(exc))
        melody_notes = parse_condition_melody(prompt_tokens, meter)
        chords = parse_condition_chords(prompt_tokens, meter)

        sample_dir = output_dir / f"{sample_index:02d}_{row['sample_id'].replace('::', '_')}"
        sample_dir.mkdir(parents=True)
        (sample_dir / "prompt-tokens.txt").write_text("\n".join(prompt_tokens) + "\n", encoding="utf-8")
        (sample_dir / "reference-tokens.txt").write_text("\n".join(reference_tokens) + "\n", encoding="utf-8")
        (sample_dir / "generated-tokens.txt").write_text("\n".join(generated_tokens) + "\n", encoding="utf-8")
        write_midi(sample_dir / "condition-melody.mid", meter, tempo, [("MELODY", melody_notes, 90)])
        write_midi(sample_dir / "condition-chords.mid", meter, tempo, [("CHORD_GUIDE", chord_notes(chords), 45)])
        write_midi(sample_dir / "reference-piano.mid", meter, tempo, [("REFERENCE_PIANO", reference_notes, 80)])
        write_midi(sample_dir / "generated-piano.mid", meter, tempo, [("GENERATED_PIANO", generated_notes, 80)])
        write_midi(sample_dir / "combined.mid", meter, tempo, [
            ("MELODY", melody_notes, 95), ("CHORD_GUIDE", chord_notes(chords), 35),
            ("GENERATED_PIANO", generated_notes, 75),
        ])
        if generated_tokens and generated_tokens[-1] == "<ARRANGEMENT_END>":
            stopping_reason = "arrangement_end"
        elif generated_tokens and generated_tokens[-1] == "<SONG_END>":
            stopping_reason = "song_end"
        elif generated_tokens.count("<BAR_END>") >= expected_bars:
            stopping_reason = "expected_bar_count"
        else:
            stopping_reason = "context_limit"
        report = {
            "sample_id": row["sample_id"], "method": METHOD, "seed": seed,
            "temperature": args.temperature, "top_k": args.top_k, "top_p": 1.0,
            "prompt_tokens": len(prompt_ids), "generated_tokens": len(generated_ids),
            "context_length": context_length, "expected_bars": expected_bars,
            "generated_bars_detected": generated_bars, "reference_bars_detected": reference_bars,
            "generated_note_count": len(generated_notes), "reference_note_count": len(reference_notes),
            "melody_note_count": len(melody_notes), "grammar_valid": not grammar_errors and generated_bars == expected_bars,
            "grammar_errors": grammar_errors, "reference_parse_errors": reference_errors,
            "stopping_reason": stopping_reason,
        }
        write_json(sample_dir / "generation-report.json", report)
        summary_rows.append(report)
        print(f"Generated {sample_index}/{len(selected)}: {row['sample_id']} valid={report['grammar_valid']}", flush=True)

    summary = {
        "status": "PASS", "method": METHOD, "model": str(model_path),
        "test_data": str(data_dir / "test.jsonl"), "selection_seed": args.selection_seed,
        "temperature": args.temperature, "top_k": args.top_k,
        "sample_count": len(summary_rows),
        "grammar_valid_count": sum(bool(row["grammar_valid"]) for row in summary_rows),
        "samples": summary_rows,
    }
    write_json(output_dir / "generation-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
