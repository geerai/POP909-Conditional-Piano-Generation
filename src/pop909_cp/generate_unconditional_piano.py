#!/usr/bin/env python3
"""Generate variable-length unconditional piano pieces with top-k sampling.

The prompt contains only global musical metadata and the Piano role.  There is
no Melody condition and no Task token.  A small grammar mask guarantees that
the sampled REMI+ stream is decodable, but it never chooses pitch, position or
duration for the model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from transformers import Qwen3ForCausalLM, set_seed


GRID_PER_QUARTER = 48
MAX_SEQUENCE_LENGTH = 2048
PIANO_VELOCITY = 80


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_vocab(path: Path) -> tuple[list[str], dict[str, int]]:
    rows = sorted(read_json(path)["tokens"], key=lambda row: int(row["id"]))
    tokens = [str(row["token"]) for row in rows]
    ids = [int(row["id"]) for row in rows]
    if ids != list(range(len(rows))) or len(tokens) != len(set(tokens)):
        raise ValueError(f"invalid vocabulary: {path}")
    if len(tokens) != 1727:
        raise ValueError(f"expected 1,727-token vocabulary, got {len(tokens)}")
    return tokens, dict(zip(tokens, ids))


def find_vocab(model_path: Path) -> Path:
    for name in (
        "remi-plus-vocab-v4.json",
        "remi-plus-vocab-v3.json",
        "remi-plus-vocab.json",
    ):
        candidate = model_path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no REMI+ vocabulary found in {model_path}")


def grids_per_bar(meter: tuple[int, int]) -> int:
    numerator, denominator = meter
    value = numerator * 4 * GRID_PER_QUARTER
    if value % denominator:
        raise ValueError(f"meter {numerator}/{denominator} is not representable")
    return value // denominator


def parse_meter(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)/(\d+)", value.strip())
    if not match:
        raise argparse.ArgumentTypeError("meter must look like 2/4 or 3/4")
    meter = (int(match.group(1)), int(match.group(2)))
    grids_per_bar(meter)
    return meter


class PianoGrammar:
    """Return legal next-token IDs for the strict single-Piano REMI+ grammar."""

    def __init__(
        self,
        id_to_token: list[str],
        token_to_id: dict[str, int],
        prompt_length: int,
        meter: tuple[int, int],
        max_notes_per_bar: int,
        max_sequence_length: int,
    ) -> None:
        self.id_to_token = id_to_token
        self.token_to_id = token_to_id
        self.prompt_length = prompt_length
        self.max_notes_per_bar = max_notes_per_bar
        self.max_sequence_length = max_sequence_length
        self.context_forced_end = False
        self.bar_size = grids_per_bar(meter)
        self.position_ids = {
            value: token_to_id[f"<POSITION_{value:03d}>"]
            for value in range(self.bar_size)
            if f"<POSITION_{value:03d}>" in token_to_id
        }
        self.pitch_ids = [
            token_to_id[f"<PITCH_{value:03d}>"]
            for value in range(128)
        ]
        self.duration_ids = [
            token_to_id[f"<DURATION_{value:03d}>"]
            for value in range(1, 769)
            if f"<DURATION_{value:03d}>" in token_to_id
        ]
        if len(self.position_ids) != self.bar_size:
            raise ValueError("vocabulary lacks positions required by the selected meter")

    def __call__(self, _batch_id: int, input_ids: torch.Tensor) -> list[int]:
        generated = [
            self.id_to_token[int(value)]
            for value in input_ids[self.prompt_length:].tolist()
        ]
        remaining = self.max_sequence_length - int(input_ids.shape[-1])
        state = "BAR_OR_END"
        completed_bars = 0
        note_count_in_bar = 0
        last_position = 0
        for token in generated:
            if state == "BAR_OR_END" and token == "<BAR>":
                state = "ROLE"
                note_count_in_bar = 0
                last_position = 0
            elif state == "BAR_OR_END" and token == "<ARRANGEMENT_END>":
                state = "SONG_END"
            elif state == "ROLE" and token == "<ROLE_PIANO>":
                state = "POSITION"
            elif state == "POSITION":
                match = re.fullmatch(r"<POSITION_(\d+)>", token)
                if not match:
                    return [self.token_to_id["<SONG_END>"]]
                last_position = int(match.group(1))
                state = "PITCH"
            elif state == "PITCH" and re.fullmatch(r"<PITCH_\d+>", token):
                state = "DURATION"
            elif state == "DURATION" and re.fullmatch(r"<DURATION_\d+>", token):
                note_count_in_bar += 1
                state = "POSITION_OR_ROLE_END"
            elif state == "POSITION_OR_ROLE_END" and token == "<ROLE_END>":
                state = "BAR_END"
            elif state == "POSITION_OR_ROLE_END":
                match = re.fullmatch(r"<POSITION_(\d+)>", token)
                if not match:
                    return [self.token_to_id["<SONG_END>"]]
                last_position = int(match.group(1))
                state = "PITCH"
            elif state == "BAR_END" and token == "<BAR_END>":
                completed_bars += 1
                state = "BAR_OR_END"
            elif state == "SONG_END" and token == "<SONG_END>":
                state = "DONE"
            else:
                return [self.token_to_id["<SONG_END>"]]

        if state == "BAR_OR_END":
            if completed_bars == 0:
                return [self.token_to_id["<BAR>"]]
            if remaining < 9:
                self.context_forced_end = True
                return [self.token_to_id["<ARRANGEMENT_END>"]]
            return [
                self.token_to_id["<BAR>"],
                self.token_to_id["<ARRANGEMENT_END>"],
            ]
        if state == "ROLE":
            return [self.token_to_id["<ROLE_PIANO>"]]
        if state == "POSITION":
            return list(self.position_ids.values())
        if state == "PITCH":
            return self.pitch_ids
        if state == "DURATION":
            return self.duration_ids
        if state == "POSITION_OR_ROLE_END":
            if note_count_in_bar >= self.max_notes_per_bar or remaining < 7:
                if remaining < 7:
                    self.context_forced_end = True
                return [self.token_to_id["<ROLE_END>"]]
            allowed = [
                token_id
                for position, token_id in self.position_ids.items()
                if position >= last_position
            ]
            if note_count_in_bar > 0:
                allowed.append(self.token_to_id["<ROLE_END>"])
            return allowed
        if state == "BAR_END":
            return [self.token_to_id["<BAR_END>"]]
        if state == "SONG_END":
            return [self.token_to_id["<SONG_END>"]]
        return [self.token_to_id["<SONG_END>"]]


def parse_complete_tokens(
    tokens: list[str], meter: tuple[int, int],
) -> tuple[list[tuple[int, int, int]], list[list[str]]]:
    expected_header = [
        "<SONG_START>",
        f"<METER_{meter[0]}_{meter[1]}>",
    ]
    if tokens[:2] != expected_header:
        raise ValueError("generated sequence has an invalid header")
    try:
        start = tokens.index("<ARRANGEMENT_START>") + 1
    except ValueError as exc:
        raise ValueError("ARRANGEMENT_START is missing") from exc
    index = start
    bar_index = 0
    bar_size = grids_per_bar(meter)
    notes: list[tuple[int, int, int]] = []
    serialized_bars: list[list[str]] = []
    while index < len(tokens) and tokens[index] == "<BAR>":
        bar_tokens = [tokens[index]]
        index += 1
        if tokens[index] != "<ROLE_PIANO>":
            raise ValueError(f"bar {bar_index + 1}: ROLE_PIANO is missing")
        bar_tokens.append(tokens[index])
        index += 1
        previous_position = -1
        while tokens[index] != "<ROLE_END>":
            triple = tokens[index:index + 3]
            if len(triple) != 3:
                raise ValueError(f"bar {bar_index + 1}: truncated note")
            position_match = re.fullmatch(r"<POSITION_(\d+)>", triple[0])
            pitch_match = re.fullmatch(r"<PITCH_(\d+)>", triple[1])
            duration_match = re.fullmatch(r"<DURATION_(\d+)>", triple[2])
            if not position_match or not pitch_match or not duration_match:
                raise ValueError(f"bar {bar_index + 1}: invalid note triple {triple}")
            position = int(position_match.group(1))
            pitch = int(pitch_match.group(1))
            duration = int(duration_match.group(1))
            if not previous_position <= position < bar_size:
                raise ValueError(f"bar {bar_index + 1}: invalid position {position}")
            previous_position = position
            notes.append((bar_index * bar_size + position, pitch, duration))
            bar_tokens.extend(triple)
            index += 3
        bar_tokens.append(tokens[index])
        index += 1
        if tokens[index] != "<BAR_END>":
            raise ValueError(f"bar {bar_index + 1}: BAR_END is missing")
        bar_tokens.append(tokens[index])
        index += 1
        serialized_bars.append(bar_tokens)
        bar_index += 1
    if bar_index < 1:
        raise ValueError("generated sequence contains no complete bar")
    if tokens[index:index + 2] != ["<ARRANGEMENT_END>", "<SONG_END>"]:
        raise ValueError("generated sequence has an invalid ending")
    return notes, serialized_bars


def write_midi(
    path: Path,
    notes: list[tuple[int, int, int]],
    meter: tuple[int, int],
    tempo: int,
) -> None:
    import miditoolkit  # type: ignore

    ppq = 480
    midi = miditoolkit.MidiFile(ticks_per_beat=ppq)
    midi.tempo_changes = [miditoolkit.TempoChange(tempo, 0)]
    midi.time_signature_changes = [
        miditoolkit.TimeSignature(meter[0], meter[1], 0)
    ]
    instrument = miditoolkit.Instrument(
        program=0,
        is_drum=False,
        name="PIANO",
    )
    instrument.notes = [
        miditoolkit.Note(
            velocity=PIANO_VELOCITY,
            pitch=pitch,
            start=round(start * ppq / GRID_PER_QUARTER),
            end=round((start + duration) * ppq / GRID_PER_QUARTER),
        )
        for start, pitch, duration in notes
    ]
    midi.instruments = [instrument]
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(path))


def repetition_rates(bars: list[list[str]]) -> tuple[float, float]:
    rendered = [tuple(bar) for bar in bars]
    if not rendered:
        return 0.0, 0.0
    duplicate_count = len(rendered) - len(set(rendered))
    adjacent = sum(left == right for left, right in zip(rendered, rendered[1:]))
    return (
        duplicate_count / len(rendered),
        adjacent / max(1, len(rendered) - 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--meter", type=parse_meter, default=(2, 4))
    parser.add_argument("--tempo", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--max-notes-per-bar", type=int, default=32)
    args = parser.parse_args()

    if not 1 <= args.sample_count <= 100:
        raise SystemExit("sample-count must be between 1 and 100")
    if not 30 <= args.tempo <= 240:
        raise SystemExit("tempo must be between 30 and 240")
    if not 1 <= args.top_k <= 100:
        raise SystemExit("top-k must be between 1 and 100")
    if args.temperature <= 0:
        raise SystemExit("temperature must be positive")
    if not 1 <= args.max_notes_per_bar <= 64:
        raise SystemExit("max-notes-per-bar must be between 1 and 64")

    vocab_path = find_vocab(args.model_path)
    id_to_token, token_to_id = load_vocab(vocab_path)
    model = Qwen3ForCausalLM.from_pretrained(
        str(args.model_path),
        dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
    )
    if model.get_input_embeddings().num_embeddings != len(id_to_token):
        raise ValueError("model and vocabulary sizes do not match")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    prompt_tokens = [
        "<SONG_START>",
        f"<METER_{args.meter[0]}_{args.meter[1]}>",
        f"<TEMPO_{args.tempo:03d}>",
        "<ROLE_PIANO>",
        "<PIANO_000>",
        "<ARRANGEMENT_START>",
    ]
    missing = [token for token in prompt_tokens if token not in token_to_id]
    if missing:
        raise ValueError(f"prompt tokens outside vocabulary: {missing}")
    prompt_ids = [token_to_id[token] for token in prompt_tokens]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for sample_index in range(args.sample_count):
        seed = args.seed + sample_index
        set_seed(seed)
        grammar = PianoGrammar(
            id_to_token,
            token_to_id,
            len(prompt_ids),
            args.meter,
            args.max_notes_per_bar,
            MAX_SEQUENCE_LENGTH,
        )
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = model.generate(
                input_ids=inputs,
                do_sample=True,
                top_k=args.top_k,
                temperature=args.temperature,
                max_new_tokens=MAX_SEQUENCE_LENGTH - len(prompt_ids),
                eos_token_id=token_to_id["<SONG_END>"],
                pad_token_id=token_to_id["<PAD>"],
                prefix_allowed_tokens_fn=grammar,
                use_cache=True,
            )[0].tolist()
        tokens = [id_to_token[value] for value in output]
        notes, bars = parse_complete_tokens(tokens, args.meter)
        sample_dir = args.output_dir / f"{sample_index + 1:02d}_seed-{seed}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "generated.tokens.txt").write_text(
            "\n".join(tokens) + "\n",
            encoding="utf-8",
        )
        write_midi(sample_dir / "generated.mid", notes, args.meter, args.tempo)
        exact_repetition, adjacent_repetition = repetition_rates(bars)
        pitches = [pitch for _start, pitch, _duration in notes]
        report = {
            "sample_number": sample_index + 1,
            "seed": seed,
            "sampling": "random top-k",
            "top_k": args.top_k,
            "temperature": args.temperature,
            "grammar_constrained": True,
            "conditioned_on_melody": False,
            "task_token_present": False,
            "meter": f"{args.meter[0]}/{args.meter[1]}",
            "tempo": args.tempo,
            "bar_count": len(bars),
            "length_selected_by_model": not grammar.context_forced_end,
            "context_forced_end": grammar.context_forced_end,
            "max_notes_per_bar": args.max_notes_per_bar,
            "note_count": len(notes),
            "pitch_min": min(pitches),
            "pitch_max": max(pitches),
            "token_count": len(tokens),
            "exact_bar_repetition_rate": exact_repetition,
            "adjacent_exact_bar_repetition_rate": adjacent_repetition,
            "midi_playable": True,
            "midi_path": str(sample_dir / "generated.mid"),
        }
        write_json(sample_dir / "generation-report.json", report)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)

    summary = {
        "model_path": str(args.model_path),
        "sample_count": len(reports),
        "generation_mode": "unconditional single-Piano random top-k sampling",
        "top_k": args.top_k,
        "temperature": args.temperature,
        "seed_start": args.seed,
        "length_policy": "model chooses ARRANGEMENT_END after any completed bar",
        "max_notes_per_bar": args.max_notes_per_bar,
        "meter": f"{args.meter[0]}/{args.meter[1]}",
        "tempo": args.tempo,
        "grammar_constrained": True,
        "mean_note_count": sum(row["note_count"] for row in reports) / len(reports),
        "minimum_bar_count": min(row["bar_count"] for row in reports),
        "maximum_bar_count": max(row["bar_count"] for row in reports),
        "mean_bar_count": sum(row["bar_count"] for row in reports) / len(reports),
        "model_selected_length_count": sum(
            row["length_selected_by_model"] for row in reports
        ),
        "mean_exact_bar_repetition_rate": sum(
            row["exact_bar_repetition_rate"] for row in reports
        ) / len(reports),
        "mean_adjacent_exact_bar_repetition_rate": sum(
            row["adjacent_exact_bar_repetition_rate"] for row in reports
        ) / len(reports),
        "all_midi_playable": all(row["midi_playable"] for row in reports),
        "samples": reports,
    }
    write_json(args.output_dir / "generation-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
