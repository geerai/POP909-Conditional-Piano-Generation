# Evaluation methodology

## Structural validity

The parser checks note triples, token types, duration fields, bar-relative
positions, position ordering, contiguous bar IDs, target endings, and stopping
reason. Completion ratio is generated bars divided by expected condition bars.

## Harmony

Chord-tone ratio (CTR) is the fraction of Piano note onsets whose pitch class
belongs to the active annotated chord. Overall CTR uses every onset;
first-beat CTR uses onsets at `POSITION_000`. Reference and generated CTR are
reported separately, with generated-minus-reference delta.

CTR is not a complete musicality score. Passing tones, suspensions, extensions,
and simplified chord annotations can create musically valid non-chord tones.

## Rhythm

- Notes per bar.
- Mean inter-onset interval over unique onset positions.
- Jaccard overlap of reference/generated onset positions in aligned bars.
- Duration histograms.

Onset Jaccard measures similarity to the reference performance, not the only
acceptable accompaniment.

## Structure

Each bar is represented by pitch-class histogram, onset histogram, and note
density. Pairwise cosine similarities form a self-similarity matrix (SSM).
Metrics include repetition rate above 0.8 similarity, SSM correlation between
reference and generation, and similarity by bar lag.

## Validity gate

Musical metrics use aligned complete bar occurrences so exploratory analysis is
possible, but invalid/missing output is never silently treated as valid. A
final model benchmark should first achieve a high structural-validity rate.

## Human evaluation next step

A blind listening study should rate harmonic fit, rhythmic naturalness,
structural coherence, playability, musical interest, and overall preference.
Samples should be stratified by CTR and grammar validity so objective/listening
disagreements can be inspected.

