# Datasets and alignment

## Sources

This project does not redistribute the source MIDI collections.

- **POP909:** supplies separately named `MELODY`, `BRIDGE`, and `PIANO`
  instrument tracks plus downbeat metadata.
- **POP909-CL:** supplies corrected chord tracks, key signatures, meters, and
  alignment information.

Songs are paired by their three-digit POP909 ID. The independent POP909 Melody
and Piano tracks are retained; Bridge is removed. The merged score track in
POP909-CL is not used as the melody condition because it can contain target
Piano material and would leak the answer.

## Time representation

Every event is converted from source ticks through that file's own PPQ to a
48-position-per-quarter grid. Original and CL files are never assumed to share
the same PPQ. Notes crossing a window boundary are normalized by the dataset
builder, and windows end only at complete bar boundaries.

## Splits

Splitting is performed by song before window creation. Windows from the same
song can therefore never appear in multiple splits.

The key/chord/melody dataset contains 2,767 train windows from 804 songs, 171
validation windows from 50 songs, and 168 held-out test windows from 49 songs.

## Audit policy

The audit verifies source presence, required tracks, key signatures, chord
events, PPQ, note ranges, timing overlap, and known alignment exceptions.
Accepted, warning, and excluded IDs are stored in `metadata/`.

The five initially excluded songs are:

- `063`: missing key signature.
- `367`: missing chord track and key signature.
- `518`: known downbeat misalignment.
- `620`: known possible hand/alignment issue.
- `658`: missing chord track.

One additional accepted song failed conversion; the converter records rather
than silently drops such failures.

## Rebuilding

Set `POP909_ORIGINAL`, `POP909_CL`, and `QWEN_MIDI_ROOT`, then run:

```bash
bash scripts/audit_pop909_cl.sh
bash scripts/prepare_key_chord_melody.sh
```

The generated JSONL, cleaned MIDI, and round-trip examples belong in a local
data directory, not in Git.

