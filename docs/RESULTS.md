# Results summary

## Training

The five-epoch 150M key/chord model reached train loss 1.1303 and validation
loss 1.3516. Extending training to 2,000 optimizer steps reduced reported train
loss to 0.5013 but increased final validation loss to 2.3699, indicating poorer
generalization at the last checkpoint.

## Paired generation comparison

For the same ten conditions and seeds, mean bar completion increased from 68.1%
to 81.4%. Recorded grammar errors changed from 22 to 21, but no sample became
fully grammar valid and five samples contained more errors after the longer
run. Longer training helped completion but did not solve target grammar.

## 100-sample diagnostic

Generated output had lower chord-tone ratios, denser onset patterns, shorter
durations, and greater repetition than the reference. SSM correlation was near
zero. All 100 sequences failed strict structural validation, so results remain
diagnostic.

Detailed machine-readable results are available under `results/`.

