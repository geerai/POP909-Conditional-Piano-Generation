# Focused project timeline

This timeline includes only stages that directly support the POP909 conditional
Piano research question.

1. **Piano-only baseline.** Clean the named POP909 Piano track, validate
   round-trip serialization, and train a 38.6M unconditional model.
2. **Melody conditioning.** Place a complete aligned Melody window before the
   Piano target and mask condition labels.
3. **Capacity expansion.** Expand the trained 12-layer model to 48 layers while
   preserving its initial function.
4. **Harmonic conditioning.** Audit POP909-CL, align Key/Chord data with the
   independent POP909 Melody and Piano tracks, and append factorized tokens.
5. **Five-epoch 150M experiment.** Train the full model on the final task.
6. **Long-run experiment.** Resume with effective batch 32 and complete 2,000
   optimizer updates without early stopping.
7. **Controlled generation comparison.** Use identical conditions, seeds, and
   REMI sampling settings for v1 and the 2,000-step model.
8. **Diagnostic evaluation.** Measure structural validity, harmony, rhythm, and
   bar-level self-similarity on 10 and 100 generated samples.

The earlier MultiPop/Jazz five-track work is a separate project. It motivated
the controlled POP909 task but is not part of this repository's implementation.
