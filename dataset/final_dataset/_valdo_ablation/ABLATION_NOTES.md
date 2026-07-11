# VALDO-2021 Haemorrhagic augmentation — negative-result ablation

## What was tried
Haemorrhagic was the most patient-thin class in the dataset (5 patients, 186 images).
VALDO-2021 Task 2 (cerebral microbleed segmentation challenge data) was extracted and
merged into the Haemorrhagic class as additional train/val-only diversity (50 new
patients, 75 T2*/GRE slices, 2 slices/patient), deliberately excluded from the test
split since microbleeds are a different, smaller, chronic lesion type than acute
hemorrhagic stroke (see `build_final_dataset.py::process_valdo`).

## Dataset composition during this run (verified directly from manifest.csv)

| | images | patients |
|---|---|---|
| Haemorrhagic total | 261 | 55 |
| — kaggle_slices (original, acute) | 186 | 5 |
| — valdo_cmb (microbleed) | 75 | 50 |

Split breakdown (images / patients):
| split | kaggle_slices | valdo_cmb |
|---|---|---|
| train | 136 / 4 | 62 / 42 |
| val | 0 / 0 | 13 / 8 |
| test | 50 / 1 | 0 / 0 |

Train set was **69% original acute-hemorrhage, 31% VALDO microbleed** by image count —
VALDO was NOT the numerical majority of training data. Test set was 100% original
acute-hemorrhage (VALDO correctly excluded), 1 patient, 50 images.

## Result (confirmed real — copied verbatim from the notebook's cell 15
## classification_report output on the VALDO-merged run)

```
Haemorrhagic    precision=0.50   recall=0.04   f1-score=0.07   support=50
```

Test recall collapsed from a pre-VALDO baseline of ~0.78 (see main manifest / prior
run) to 0.04 — the model almost never correctly identified genuine acute Haemorrhagic
test cases after VALDO was added to training.

**Confusion matrix from this run: not available.** The notebook file's saved outputs
were cleared before this was noticed (every edit in this project's workflow clears
cell outputs), so the actual "predicted as X" breakdown for the 50 test cases has not
been captured. If this run is reproduced, capture cell 14's confusion matrix output
before re-editing the notebook.

## Diagnosis
Not a data-quantity problem (VALDO was a minority of train images). The more likely
mechanism: forcing one decision boundary to cover two visually disjoint pathologies —
tiny punctate chronic microbleeds vs. large acute hemorrhagic-stroke bleeds — under a
single "Haemorrhagic" label degrades the model's ability to recognize either well,
even when one sub-type is technically the numerical majority. This is a class
heterogeneity / label-granularity problem, not a class-imbalance problem.

## Decision
VALDO is excluded from the default `final_dataset/` build (see
`build_final_dataset.py`'s `INCLUDE_VALDO` flag, default `False`). The extraction
code (`process_valdo`) is kept fully intact and runnable — set `INCLUDE_VALDO = True`
to reproduce this exact ablation. The 75 VALDO-derived images and their manifest
fragment are preserved in this folder (`_valdo_ablation/`) rather than deleted, for
exactly this purpose: this is a documented negative result, not a discarded mistake.

## For the paper
Suggested framing (Discussion / ablations section): "We evaluated augmenting the
critically patient-thin Haemorrhagic class (5 patients) with VALDO-2021 cerebral
microbleed data (+50 patients, 75 images, train/val only). Despite VALDO representing
a minority (31%) of Haemorrhagic training images, test recall for acute hemorrhagic
stroke collapsed from ~0.78 to 0.04 (precision 0.50, F1 0.07, n=50). We attribute this
to class heterogeneity: microbleeds and acute hemorrhagic stroke, while both
hemorrhage-related, are visually and clinically distinct pathologies, and merging them
under one label appears to degrade the decision boundary for both rather than simply
diluting it proportionally to their representation. This suggests that domain-specific
data augmentation for stroke MRI classification requires lesion-type matching, not
just modality/sequence matching."
