# Data card

## Data sources

- POP909: independent Melody and Piano tracks and timing metadata.
- POP909-CL: corrected Key, Chord, meter, and alignment annotations.

The repository does not redistribute either source corpus.

## Construction

Songs are paired by POP909 ID. Bridge is removed. The original Melody track is
the condition and the original Piano track is the target. POP909-CL supplies
the harmonic annotations. Each source uses its own PPQ-to-grid conversion.

Windows contain aligned complete bars, a maximum of 32 bars, and at most 2,048
tokens. Splits are disjoint at song level.

## Final split

| Split | Songs | Windows |
|---|---:|---:|
| Train | 804 | 2,767 |
| Validation | 50 | 171 |
| Test | 49 | 168 |

The audit accepted 904 source songs; 903 passed conversion. Exclusion reasons
and accepted IDs are versioned in `metadata/`.

## Risks and caveats

- Chord annotations may simplify extensions, suspensions, or passing harmony.
- The dataset represents one pop-music collection and should not be assumed to
  cover all accompaniment styles.
- Multiple windows can originate from one song inside a split; sample-level
  confidence intervals must account for song correlation in a final study.
- Reference Piano is one valid accompaniment, not the unique correct target.

