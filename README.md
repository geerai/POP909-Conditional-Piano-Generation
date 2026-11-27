# POP909 Conditional Piano Generation

Symbolic piano accompaniment generation with a Qwen3-style decoder-only
Transformer. The current task is:

> **Given key, chord progression, and melody, generate a complete piano part.**

The repository follows one controlled research progression:

1. POP909 piano-only unconditional language modeling.
2. Melody-conditioned piano generation.
3. Function-preserving expansion from 38.6M to 151.96M parameters.
4. Key + chord + melody-conditioned piano generation using POP909-CL.
5. Structural, harmonic, rhythmic, and self-similarity evaluation.

[Listen to the interactive demos](https://geerai.github.io/Demo-Page/)

## Research scope

The central question is whether explicit musical conditions improve controllable
piano accompaniment generation. Earlier MultiPop/Jazz five-track experiments
motivated this narrower task, but their code is intentionally kept in a
[separate project](https://github.com/geerai/Track-Atlas-Demo-Page). This
repository contains only experiments that directly support the POP909
conditional-piano study.

## Model

| Property | Initial model | Expanded model |
|---|---:|---:|
| Architecture | Qwen3-style decoder-only Transformer | Same |
| Parameters | ~38.6M | 151,963,136 |
| Layers | 12 | 48 |
| Hidden size | 512 | 512 |
| Intermediate size | 1,536 | 1,536 |
| Attention / KV heads | 8 / 4 | 8 / 4 |
| Context | 2,048 atomic tokens | 2,048 atomic tokens |
| Vocabulary | 1,727-token REMI+ v4 | 1,782 tokens after key/chord expansion |
| Trainable parameters | All | All |
| LoRA / QLoRA | None | None |

The depth migration copies all trained parameters and inserts identity-like
residual blocks. The vocabulary migration preserves every old token ID and
every old embedding/output row. Both migrations include numerical equivalence
checks before saving.

## Data

The pipeline combines two public sources by song ID:

- [POP909](https://github.com/music-x-lab/POP909-Dataset): independent
  `MELODY` and `PIANO` tracks.
- [POP909-CL](https://github.com/AndyWeasley2004/POP909-CL-Dataset): corrected
  key, chord, meter, and alignment information.

`BRIDGE` is excluded. Melody is the condition and Piano is the target. Each
source MIDI is converted through its own PPQ into a 48-position-per-quarter
grid, which avoids assuming that all files share the same tick resolution.

The audited key/chord dataset contains:

| Split | Songs | Windows |
|---|---:|---:|
| Train | 804 | 2,767 |
| Validation | 50 | 171 |
| Test | 49 | 168 |

Five known-problem songs are excluded by the audit. One additional accepted
song failed conversion, leaving 903 successfully converted songs. Splits are
made by song, never by window.

Raw and derived corpora are not redistributed here. The preparation scripts,
split metadata, audit reports, and base vocabulary are versioned so the data
can be rebuilt from the original sources.

## Tokenization

Every bracketed element is one atomic token. The final condition is serialized
bar by bar:

```text
<SONG_START>
<TASK_KEY_CHORD_MELODY_TO_PIANO>
<METER_4_4>
<TEMPO_100>
<CONDITION_START>

<BAR_000>
<KEY_ROOT_C> <KEY_MODE_MAJOR>
<ROLE_CHORD>
<POSITION_000> <CHORD_ROOT_C> <CHORD_QUALITY_M> <CHORD_BASS_C>
<POSITION_096> <CHORD_ROOT_G> <CHORD_QUALITY_D7> <CHORD_BASS_B>
<ROLE_END>
<ROLE_MELODY>
<POSITION_000> <PITCH_064> <DURATION_024>
<ROLE_END>
<BAR_END>

<CONDITION_END>
<ARRANGEMENT_START>
<BAR_000>
<ROLE_PIANO>
<POSITION_000> <PITCH_048> <DURATION_048>
<ROLE_END>
<BAR_END>
<ARRANGEMENT_END>
<SONG_END>
```

Key and chord symbols are factorized into root, quality, and bass tokens.
Enharmonic spellings are normalized to 12 pitch classes. Condition and target
always use the same relative bar IDs.

Only target tokens after `<ARRANGEMENT_START>` contribute to the loss. Key,
chord, and melody tokens remain visible as context but their labels are masked
to `-100`.

See [Tokenization](docs/TOKENIZATION.md) for the full specification.

## Training

All stages use teacher forcing and ordinary causal cross-entropy. The main
150M experiment uses two-GPU Distributed Data Parallel training, full-parameter
optimization, AdamW, constant learning rate after warmup, and gradient clipping
at 1.0.

| Run | Optimizer budget | Train loss | Validation loss | Notes |
|---|---:|---:|---:|---|
| Unconditional Piano 38M | up to 20 epochs | 0.639 final log | 1.0906 best | Early stopping enabled |
| Melody → Piano 38M | 5 epochs reached | — | 1.1945 best checkpoint evaluation | Conditional baseline |
| Key + Chord + Melody 150M v1 | 5 epochs | 1.1303 | 1.3516 | All epochs completed |
| Key + Chord + Melody 150M long | 2,000 optimizer steps | 0.5013 | 2.3699 final | No early stopping; overfitting signal |

The long run was deliberately allowed to finish all 2,000 optimizer steps. Its
lower training loss did not translate to lower validation loss, so training
duration alone should not be treated as improvement.

See [Training methodology](docs/TRAINING.md) and the exact configurations in
[`configs/`](configs/).

## Generation

The reported key/chord/melody runs use unconstrained REMI sampling:

```text
temperature = 1.2
top_k = 5
top_p = 1.0
grammar constraint = false
```

Generation receives the complete serialized test condition and produces only
the Piano target. Each sample folder stores condition Melody, condition Chords,
reference Piano, generated Piano, combined MIDI, tokens, and a machine-readable
report.

## Evaluation

The evaluator compares generated Piano with the aligned reference using:

- **Grammar:** valid-rate, error categories, stopping reason, and bar completion.
- **Harmony:** overall and first-beat chord-tone ratio (CTR).
- **Rhythm:** notes per bar, mean onset IOI, onset Jaccard, and durations.
- **Structure:** bar-level self-similarity matrices, repetition rate, SSM
  correlation, and lag similarity.

The 100-sample diagnostic set found:

- 0/100 strictly grammar-valid token sequences;
- 77.3% mean reported bar completion;
- generated overall CTR 42.35% versus reference 66.56%;
- generated first-beat CTR 43.87% versus reference 63.36%;
- denser, shorter-note output than the reference;
- generated repetition rate 70.09% versus reference 50.07%;
- mean reference/generated SSM correlation 0.029.

These are diagnostic findings, not a final benchmark. Invalid or missing bars
limit what can be concluded about complete-song musical quality.

In a paired 10-sample comparison, the 2,000-step model increased mean bar
completion from 68.1% to 81.4% and reduced total recorded grammar errors from
22 to 21. However, neither model produced a fully valid sequence, and 5/10
samples had more errors after the longer training. The result is therefore
mixed rather than a categorical improvement.

See [Evaluation methodology](docs/EVALUATION.md) and [`results/`](results/).

## Repository layout

```text
configs/          Exact hyperparameters and sampling settings
docs/             Data, tokenization, training, generation, and evaluation notes
examples/         Paired v1 and 2,000-step generations for the same conditions
metadata/         Dataset audits, accepted/excluded songs, and POP909 index
results/          Compact 10/100-sample evaluation and grammar comparison
scripts/          Reproducible preparation, training, generation, and evaluation
src/pop909_cp/    Python implementation
vocab/            Fixed 1,727-token REMI+ v4 base vocabulary
```

## Installation

```bash
git clone https://github.com/geerai/POP909-Conditional-Piano-Generation.git
cd POP909-Conditional-Piano-Generation
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Set the workspace and dataset locations before using the launchers:

```bash
export QWEN_MIDI_ROOT=/path/to/qwen-midi
export POP909_ORIGINAL=/path/to/POP909-original
export POP909_CL_ROOT=/path/to/POP909-CL-Dataset
export POP909_CL_PROCESSED=/path/to/POP909-CL-Dataset/POP909_processed
```

## Reproduce the pipeline

```bash
# 1. Unconditional POP909 Piano baseline
bash scripts/prepare_unconditional.sh
bash scripts/train_unconditional_2gpu.sh

# 2. Melody-conditioned baseline
bash scripts/prepare_melody_to_piano.sh
bash scripts/train_melody_to_piano_2gpu.sh

# 3. Audit and build the key/chord/melody dataset
bash scripts/audit_pop909_cl.sh
bash scripts/prepare_key_chord_melody.sh

# 4. Expand depth and vocabulary
bash scripts/expand_model.sh

# 5. Train the 150M conditional model
bash scripts/train_key_chord_5epoch_2gpu.sh

# 6. Generate and evaluate
NUM_SAMPLES=100 bash scripts/generate_key_chord.sh
INPUT_DIR=/path/to/generation-run bash scripts/evaluate_key_chord.sh
```

Model weights and optimizer checkpoints are intentionally not stored in
ordinary Git. See [Reproducibility](docs/REPRODUCIBILITY.md) for the artifacts
still required to reproduce a trained checkpoint exactly.

## Current limitations

- Generation is not grammar constrained, and bar IDs can be duplicated,
  skipped, or emitted out of order.
- The 2,048-token context often truncates long conditions/targets.
- The model underuses supplied chord tones and overproduces short notes.
- The current objective metrics require human listening validation.
- No embedding-based FID/FMD result is claimed yet.

## Project history

The earlier MultiPop/Jazz work studied heterogeneous five-track generation and
instrument-role normalization. It is useful context for why this project moved
to a more controlled, evaluable single-target task, but it is outside this
repository's research scope. See the
[Track Atlas repository](https://github.com/geerai/Track-Atlas-Demo-Page) for
that separate line of work.

