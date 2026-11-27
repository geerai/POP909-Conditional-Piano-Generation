# Tokenization specification

## Shared representation

- Every `<...>` item is one atomic token.
- Time resolution is 48 grid positions per quarter note.
- A note is `POSITION -> PITCH -> DURATION`.
- Maximum sequence length is 2,048 tokens.
- Windows contain complete bars only.
- Empty aligned bars are retained.

## Stage 1: Piano unconditional

```text
<SONG_START>
<METER_4_4>
<TEMPO_100>
<ROLE_PIANO>
<PIANO_000>
<ARRANGEMENT_START>
<BAR>
<ROLE_PIANO>
<POSITION_000> <PITCH_048> <DURATION_048>
<ROLE_END>
<BAR_END>
<ARRANGEMENT_END>
<SONG_END>
```

Only the explicitly named POP909 Piano track is present.

## Stage 2: Melody to Piano

The full melody window appears before the full aligned target window:

```text
header
<CONDITION_START>
melody bars
<CONDITION_END>
<ARRANGEMENT_START>
piano bars
<ARRANGEMENT_END>
```

This layout lets every Piano token attend to the complete melody condition.

## Stage 3: Key + Chord + Melody to Piano

Each condition bar is assigned a relative ID (`BAR_000` through `BAR_031`) and
contains Key, Chord, and Melody in a fixed order. Chords are positioned events;
their duration is implicit until the next chord event.

Chord symbols are factorized into:

- 12 chord roots;
- 13 supported qualities;
- 12 bass pitch classes;
- `CHORD_OTHER` and `NO_CHORD` fallbacks.

Keys use 12 roots and Major/Minor mode. Enharmonic spellings map to a common
pitch class.

## Vocabulary preservation

The fixed base vocabulary has 1,727 tokens. Key/chord tokens are appended, not
inserted, yielding 1,782 tokens. All legacy token IDs remain unchanged. Run the
dataset builder to emit `remi-plus-vocab-key-chord-v1.json`, then use
`expand_qwen3_vocab.py` to add matching model rows without changing old-token
logits.

## Loss boundary

`loss_start_token_index` points to the first target bar after
`ARRANGEMENT_START`. The collator copies input IDs to labels, then replaces all
labels before this index with `-100`. Prompt tokens condition the model but do
not contribute directly to cross-entropy.

