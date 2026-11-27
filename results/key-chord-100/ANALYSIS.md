# Exploratory evaluation: 100 REMI-sampling generations

This report evaluates the 100-sample archive before any of the samples are
added to the GitHub demo. The results are diagnostic, not yet a final model
benchmark, because every generated token sequence has at least one structural
error.

## 1. Evaluation gate: generation validity

- Samples in archive: **100**
- Unique test records: **100**
- Unique POP909 songs: **45**
- Grammar-valid generations: **0/100**
- Generations ending with a complete `<ROLE_END> <BAR_END>`: **48/100**
- Stopping reasons: **49 context limit**, **46 expected bar count**, **3 song
  end**, **2 arrangement end**
- Mean reported bar completion: **77.3%**; median: **81.0%**
- Paired complete bars used for musical metrics: **1,847**
- Mean paired-bar coverage: **73.6%**
- Strictly clean generations with contiguous bar IDs, no duplicates, no
  malformed note positions, and a complete ending: **0/100**

Most failures are not unreadable MIDI files. The main problem is that emitted
`BAR_xxx` IDs are skipped, duplicated, or out of order. Twelve samples also
contain malformed note triples, and seven contain positions outside the meter's
bar length. For example, a 3/4 sample emits `POSITION_144`, although valid 3/4
positions are below 144 on the 48-grid.

Therefore, the musical metrics below retain only the first complete occurrence
of each generated bar ID and compare it with the condition/reference bar having
the same ID. This makes an exploratory comparison possible, but it does not
erase the generation-validity failure.

## 2. Harmony: chord-tone ratio (CTR)

CTR tests every piano note onset against the active annotated chord. A note is
a hit when its pitch class is one of the chord tones. `First-beat CTR` applies
the same test only to notes at `POSITION_000`.

| Metric | Reference | Generated | Generated - Reference |
|---|---:|---:|---:|
| Overall CTR, macro mean | 66.56% | 42.35% | **-24.21 pp** |
| First-beat CTR, macro mean | 63.36% | 43.87% | **-18.85 pp** |
| Overall CTR, micro | 65.83% | 42.73% | -23.11 pp |
| First-beat CTR, micro | 63.10% | 43.32% | -19.78 pp |

Sample-level bootstrap 95% confidence intervals:

- Overall CTR delta: **-28.80 to -19.58 percentage points**
- First-beat CTR delta: **-25.31 to -12.47 percentage points**

Interpretation: on the aligned, parseable bars, generated piano notes are much
less likely than reference notes to belong to the supplied chord. This is the
clearest current quality problem. It should be confirmed by listening to a
stratified subset because chord annotations can sometimes be incomplete or
musically simplified.

## 3. Rhythm

| Metric | Reference | Generated |
|---|---:|---:|
| Mean notes per bar | 11.95 | **15.50** |
| Mean unique-onset IOI, 48-grid units | 23.70 | **19.99** |
| Mean aligned-bar onset Jaccard | - | **0.495** |

The generated piano is denser and places unique onsets closer together. An
onset Jaccard of about 0.50 means that only roughly half of the union of onset
positions is shared with the reference in corresponding bars. Exact agreement
is not required for a valid accompaniment, so Jaccard is a style/alignment
diagnostic rather than a quality score.

Duration distribution:

| Duration in 48-grid units | Reference | Generated |
|---|---:|---:|
| <= 12 | 2,555 | **6,508** |
| 13-24 | 4,336 | **8,300** |
| 25-48 | 8,385 | 7,642 |
| 49-96 | 4,786 | 4,268 |
| > 96 | **1,521** | 382 |

The model strongly overproduces short notes and rarely produces long notes.
This should be audible as a busier, more detached accompaniment.

## 4. Structure

Each bar is represented by a normalized pitch-class histogram, a 16-bin onset
histogram, and note density. Cosine similarity between every pair of bars forms
the self-similarity matrix (SSM).

| Metric | Reference | Generated |
|---|---:|---:|
| Repetition rate at SSM similarity >= 0.8 | 50.07% | **70.09%** |
| Mean reference/generated SSM correlation | - | **0.029** |

The 95% bootstrap interval for SSM correlation is **-0.002 to 0.061**. The
generated output is substantially more repetitive, and its large-scale
similarity pattern has almost no correlation with the reference. SSM
correlation is diagnostic only: a new accompaniment need not copy the exact
reference arrangement.

## 5. What can and cannot be claimed

These 100 samples are enough for a first diagnostic comparison of this sampling
configuration (`temperature=1.2`, `top_k=5`). They are not yet suitable for a
final benchmark because:

1. all 100 token sequences fail structural validation;
2. only 45 songs are represented, so windows from the same song are correlated;
3. invalid/missing bars make complete-song structure scores unreliable;
4. FID/FMD requires a fixed embedding model and a larger independent reference
   set; it should not be improvised from these 100 samples alone.

## 6. Recommended next experiment

1. Fix or constrain generation so bar IDs are forced to be contiguous and
   positions are valid for the current meter.
2. Reserve enough context for the target, or generate one bar at a time while
   retaining the condition, so outputs do not terminate at 2048 tokens.
3. Regenerate the same 100 test records with the same seeds. This creates a
   controlled before/after comparison.
4. Re-run the current metrics and require a high validity rate before treating
   harmony/rhythm/structure as formal results.
5. Listen to 20 stratified samples: five high-CTR, five low-CTR, five near the
   reference CTR, and five where objective metrics disagree. Rate chord fit,
   rhythm naturalness, structure, and overall musicality on a 1-5 scale.

## Output files

- `evaluation-summary.csv`: one row per sample
- `dataset-summary.json`: aggregate metrics and method definitions
- `evaluation-details.json`: full sample- and bar-level results
- `aggregate-duration-histogram.csv`: reference/generated duration counts
- `aggregate-lag-similarity.csv`: SSM similarity by bar lag
- Each sample folder: `evaluation.json`, per-bar harmony, and both SSM matrices
