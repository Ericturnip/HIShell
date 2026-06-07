# Presentation Visuals: HIShells PV Project

Generated figures live in this folder. Each PNG is slide-ready.

## 01_pipeline_overview.png

Shows the end-to-end idea: raw HI cubes are converted into physically standardized PV cuts, labeled with beam-aware masks, passed through a high-recall U-Net, then cleaned with deterministic connected-component filters.

Main talking point: the neural network is not the final catalog. It is the sensitive detection layer; physics-aware filtering and human review turn detections into credible candidates.

## 02_shell_type_schematic.png

Explains the three catalog shell types in simple PV-space language:

- Type 1: both sides stalled, or no clear velocity caps.
- Type 2: one side expanding.
- Type 3: both sides expanding.

Main talking point: type classification is a natural last-stage helper for reviewers because it tells them what PV signature to expect.

## 03_standardized_pv_label_examples.png

Shows actual standardized model inputs: each PV image is resampled to a fixed physical shape, with catalog label contours overlaid.

Main talking point: every input has the same physical meaning, not just the same pixel dimensions: 5 kpc across and 200 km/s in velocity, mapped into a 96 x 256 tensor.

## 04_dataset_composition.png

Summarizes how many PV cuts are in each split and which galaxies contribute the most cuts.

Main talking point: the model is trained on many orientations and offsets around catalog shells, plus background/random cuts, while NGC 3031 is isolated as a stress split.

## 05_training_curves.png

Shows the clean physical baseline training run.

Main talking point: the model is intentionally optimized for recall. That means it is allowed to be over-inclusive at the pixel level because later post-processing removes obvious false positives.

## 06_postprocessing_metrics.png

Shows before/after patch precision and recall at threshold 0.075.

Key numbers:

- Test after beam-area filtering: precision = 0.826, recall = 0.999.
- Stress/NGC 3031 after beam-area filtering: precision = 0.753, recall = 0.976.

Main talking point: beam-aware filtering improves precision while preserving near-perfect recall on validation/test.

## 07_probability_ranking_curve.png

Shows what happens when candidates are sorted by integrated probability mass and reviewers inspect only the top N.

Main talking point: ranking gives a tunable review workload. You can choose a smaller high-confidence list or a larger high-recall list.

## 08_ngc3031_stress_test.png

Shows why NGC 3031/M81 behaves differently: it produces far more edge-touching components than normal validation/test splits.

Main talking point: NGC 3031 is not just a file-format problem. It contains line-of-sight confusion and tidal structures, so it is best treated as a stress test for post-processing.

## Suggested slide order

1. Problem: detecting HI shells in PV space.
2. `01_pipeline_overview.png`
3. `02_shell_type_schematic.png`
4. `03_standardized_pv_label_examples.png`
5. `04_dataset_composition.png`
6. `05_training_curves.png`
7. `06_postprocessing_metrics.png`
8. `07_probability_ranking_curve.png`
9. `08_ngc3031_stress_test.png`
10. Conclusion: high recall first, physics-aware cleanup second, human review last.

## Generated Files

- `01_pipeline_overview.png`
- `02_shell_type_schematic.png`
- `03_standardized_pv_label_examples.png`
- `04_dataset_composition.png`
- `05_training_curves.png`
- `06_postprocessing_metrics.png`
- `07_probability_ranking_curve.png`
- `08_ngc3031_stress_test.png`
