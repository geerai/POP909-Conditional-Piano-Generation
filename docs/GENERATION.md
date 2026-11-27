# Generation protocol

## Conditions

Generation uses held-out test records. The complete Key/Chord/Melody condition
is retained and the paired reference Piano is never included in the prompt.
Only tokens through `ARRANGEMENT_START` are supplied to the model.

## Reported REMI sampling

The principal comparison uses temperature 1.2, top-k 5, top-p 1.0, identical
selection and generation seeds, and no grammar constraint. Keeping the same
conditions and seeds makes the five-epoch and 2,000-step outputs paired.

## Outputs

Each sample directory stores:

- `condition-melody.mid`;
- `condition-chords.mid`;
- `reference-piano.mid`;
- `generated-piano.mid`;
- `combined.mid`;
- prompt/reference/generated token text;
- `generation-report.json`.

## Stopping

Generation can stop at the expected number of condition bars, an arrangement
ending token, a song ending token, or the 2,048-token context limit. The report
preserves the stopping reason and all parser errors.

## Future comparisons

Random top-k and beam/grid search should use the same test conditions and
evaluation code. Grammar-constrained decoding should be reported as a distinct
method rather than mixed with the unconstrained baseline.

