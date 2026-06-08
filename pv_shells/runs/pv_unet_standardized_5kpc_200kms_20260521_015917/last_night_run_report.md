# Last Night Standardized U-Net Run Report

Run directory:

`pv_shells/runs/pv_unet_standardized_5kpc_200kms_20260521_015917/`

Report written: 2026-05-21

No training was launched while preparing this report. I only inspected artifacts and ran inference-only evaluation on the saved checkpoints.

## 1. Run Artifacts Found

| artifact | timestamp | note |
|---|---:|---|
| `high_recall_model.keras` | 2026-05-21 02:02:54 | Saved very early; behaves like maximum-recall / maximum-overplot checkpoint. |
| `best_model.keras` | 2026-05-21 02:10:25 | Best practical checkpoint from this run; likely selected by validation PR-AUC / validation loss around epoch 2. |
| `final_model.keras` | 2026-05-21 02:57:16 | Final epoch checkpoint after validation quality had degraded from the early peak. |
| `history.csv` | 2026-05-21 02:57:16 | Full training history. |
| `history_final.json` | 2026-05-21 02:57:16 | Same training history in JSON form. |
| `eval_val_best_model.json` | 2026-05-21 13:14:03 | Inference-only validation evaluation generated for this report. |
| `eval_test_best_model.json` | 2026-05-21 13:15:31 | Inference-only test evaluation generated for this report. |
| `eval_val_high_recall_model.json` | 2026-05-21 13:16:32 | Inference-only validation evaluation generated for this report. |
| `eval_test_high_recall_model.json` | 2026-05-21 13:18:03 | Inference-only test evaluation generated for this report. |

## 2. Dataset / Config Used

Training config:

`pv_shells/training_data/standardized_5kpc_200kms/train_standardized_high_recall.yaml`

The config points at the standardized manifest set:

| split | manifest |
|---|---|
| train | `pv_shells/training_data/standardized_5kpc_200kms/train_manifest.csv` |
| validation | `pv_shells/training_data/standardized_5kpc_200kms/val_manifest.csv` |
| test | `pv_shells/training_data/standardized_5kpc_200kms/test_manifest.csv` |

Standardized representation:

| field | value |
|---|---:|
| spatial window | 5.0 kpc |
| velocity window | 200.0 km/s |
| velocity centering | local Moment-1 velocity |
| training patch shape | 256 spatial x 96 velocity |
| samples per PV cut | 2 |
| label cleaning mode | `audit_only` |
| loss | `bce_tversky` |
| Tversky alpha / beta | 0.3 / 0.7 |
| model selection monitor | `val_pr_auc` |
| high-recall monitor | `val_patch_recall_0p075` |

The split summary reports galaxy-held-out splitting:

| split | manifest rows | positives | negatives | galaxies |
|---|---:|---:|---:|---|
| train | 23,784 | 17,400 | 6,384 | `ngc_628`, `ngc_2403`, `ngc_2841`, `ngc_2976`, `ngc_3031`, `ngc_3521`, `ngc_3627`, `ngc_4214`, `ddo154`, `ngc_5194`, `ngc_6946` |
| validation | 2,660 | 1,779 | 881 | `ngc_2366`, `ho_i`, `ngc_4449`, `ngc_4736` |
| test | 3,680 | 2,547 | 1,133 | `ddo53`, `ho_ii`, `ngc_3184`, `ngc_7793` |

Because `samples_per_pv: 2`, the aggregate evaluator saw 5,320 validation patches and 7,360 test patches.

## 3. Validation Realism

The standardized split is much more deployment-realistic than the earlier centered-cut-only workflow. The validation set includes centered positives, spatial offsets, angle offsets, velocity offsets, random nearby/grazing cuts, fine-grid deployment-like cuts, and background negatives.

Validation category counts from `standardized_split_summary.json`:

| validation cut category | count | positives | negatives |
|---|---:|---:|---:|
| centered positives, all four angle definitions combined | 304 | 304 | 0 |
| spatial-offset positives, all offset categories combined | 912 | 682 | 230 |
| angle-offset positives, all angle offsets combined | 304 | 304 | 0 |
| velocity-offset cuts, all offsets combined | 304 | 303 | 1 |
| random nearby / grazing | 152 | 152 | 0 |
| fine-grid deployment-like | 380 | 22 | 358 |
| background random negative | 304 | 12 | 292 |

This is a meaningful improvement, but the validation set is still dominated by catalog-associated positive cut families. The fine-grid deployment-like portion is present and useful, but category-wise model evaluation is still needed to answer whether failures are concentrated in grid/background cuts or offset/grazing cuts.

## 4. Training Curve Summary

The run completed 15 epochs, from epoch 0 through epoch 14, despite the config allowing 80 epochs. This is consistent with early stopping / restored best checkpoint behavior.

Key history points:

| epoch | train PR-AUC | val PR-AUC | val loss | val pixel recall @0.075 | val pixel precision @0.075 | val patch recall @0.075 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1843 | 0.1177 | 0.7456 | 0.9451 | 0.0713 | 1.0000 |
| 1 | 0.2484 | 0.2257 | 0.3747 | 0.3880 | 0.3665 | 0.9679 |
| 2 | 0.2598 | 0.2670 | 0.3542 | 0.6111 | 0.2534 | 1.0000 |
| 5 | 0.3020 | 0.2596 | 0.3817 | 0.3779 | 0.3581 | 0.9725 |
| 10 | 0.3671 | 0.1603 | 0.4085 | 0.1914 | 0.4626 | 0.8595 |
| 14 | 0.3808 | 0.2006 | 0.3743 | 0.3044 | 0.4123 | 0.9267 |

Interpretation:

- Validation PR-AUC and validation loss both peaked around epoch 2.
- Training PR-AUC continued improving through epoch 14.
- This points to early overfitting, a train/validation distribution mismatch, or both.
- The final checkpoint should not be treated as the best operating checkpoint.
- The high-recall checkpoint appears to be from epoch 0; it has extreme recall but extremely poor review-load behavior.

## 5. Aggregate Inference Evaluation

The following evaluations used `scripts/evaluate_pv_unet.py` with the standardized config. The TensorFlow environment printed a warning that `astropy` was unavailable and the dataset loader used the raw training config. Since the config contains absolute manifest paths and evaluation completed, this does not invalidate the results, but it should be fixed for cleaner future runs.

### Best Model

Checkpoint:

`pv_shells/runs/pv_unet_standardized_5kpc_200kms_20260521_015917/best_model.keras`

| split | patches | positives | negatives | loss | pixel PR-AUC |
|---|---:|---:|---:|---:|---:|
| validation | 5,320 | 3,558 | 1,762 | 0.0510 | 0.2449 |
| test | 7,360 | 5,094 | 2,266 | 0.0792 | 0.2880 |

Threshold metrics:

| split | threshold | pixel precision | pixel recall | pixel F1 | patch precision | patch recall | patch F1 | patch FP | patch FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.05 | 0.2189 | 0.6391 | 0.3261 | 0.7820 | 0.9938 | 0.8752 | 986 | 22 |
| validation | 0.075 | 0.2237 | 0.6301 | 0.3301 | 0.7861 | 0.9938 | 0.8779 | 962 | 22 |
| validation | 0.1 | 0.2271 | 0.6238 | 0.3329 | 0.7875 | 0.9938 | 0.8787 | 954 | 22 |
| test | 0.05 | 0.2616 | 0.6516 | 0.3733 | 0.8176 | 0.9996 | 0.8995 | 1,136 | 2 |
| test | 0.075 | 0.2664 | 0.6451 | 0.3771 | 0.8210 | 0.9996 | 0.9016 | 1,110 | 2 |
| test | 0.1 | 0.2698 | 0.6404 | 0.3797 | 0.8221 | 0.9996 | 0.9022 | 1,102 | 2 |

Best-model interpretation:

- This checkpoint is aligned with the high-recall localizer philosophy.
- Patch recall is excellent at all three operating thresholds.
- Pixel recall is moderate, around 0.63-0.65 at 0.05-0.075.
- Pixel precision remains low, which is acceptable for this stage as long as deterministic component filtering follows.
- The difference between 0.05, 0.075, and 0.1 is small for this checkpoint. Threshold 0.075 remains the best balanced high-recall operating point, while 0.05 is still useful when maximum recall is desired.

### High-Recall Checkpoint

Checkpoint:

`pv_shells/runs/pv_unet_standardized_5kpc_200kms_20260521_015917/high_recall_model.keras`

| split | patches | positives | negatives | loss | pixel PR-AUC |
|---|---:|---:|---:|---:|---:|
| validation | 5,320 | 3,558 | 1,762 | 0.5222 | 0.0977 |
| test | 7,360 | 5,094 | 2,266 | 0.6384 | 0.1206 |

Threshold metrics:

| split | threshold | pixel precision | pixel recall | pixel F1 | patch precision | patch recall | patch F1 | patch FP | patch FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.05 | 0.0523 | 0.9565 | 0.0992 | 0.6688 | 1.0000 | 0.8015 | 1,762 | 0 |
| validation | 0.075 | 0.0534 | 0.9544 | 0.1012 | 0.6688 | 1.0000 | 0.8015 | 1,762 | 0 |
| validation | 0.1 | 0.0542 | 0.9530 | 0.1026 | 0.6688 | 1.0000 | 0.8015 | 1,762 | 0 |
| test | 0.05 | 0.0734 | 0.9252 | 0.1361 | 0.6923 | 1.0000 | 0.8182 | 2,264 | 0 |
| test | 0.075 | 0.0748 | 0.9225 | 0.1384 | 0.6923 | 1.0000 | 0.8182 | 2,264 | 0 |
| test | 0.1 | 0.0758 | 0.9204 | 0.1401 | 0.6923 | 1.0000 | 0.8182 | 2,264 | 0 |

High-recall interpretation:

- This checkpoint effectively flags almost every patch.
- It achieves perfect patch recall but produces nearly all available negative patches as false positives.
- It is useful as an upper-bound sanity check for recall, but it is not a practical checkpoint for human review or candidate harvesting.

## 6. Threshold Recommendation

For `best_model.keras`:

- Use threshold `0.075` as the default high-recall operating point.
- Keep threshold `0.05` as maximum-recall mode for review panels and candidate harvesting.
- Threshold `0.1` gives slightly cleaner masks but only a small reduction in false-positive patch load for this run.

Because patch recall is nearly saturated for the best checkpoint, the next useful diagnostic is not another aggregate threshold sweep. The next useful diagnostic is category-wise performance: centered vs spatial-offset vs velocity-offset vs grazing vs fine-grid deployment-like vs background.

## 7. Main Concerns

1. Early overfitting / distribution mismatch

   The best validation metrics arrive around epoch 2, while training metrics continue improving. This suggests the model quickly starts fitting the training distribution in ways that do not transfer cleanly to the held-out galaxies.

2. Fine-grid behavior is not isolated yet

   The validation/test splits include deployment-like fine-grid cuts, but the current aggregate evaluator does not break metrics down by cut category. The aggregate patch recall may hide failures on blind-grid cuts because catalog-associated positive families dominate the positive examples.

3. High-recall checkpoint is too broad

   The high-recall checkpoint is not just recall-friendly; it is nearly all-positive at the patch level. It should not be the main checkpoint for downstream candidate review.

4. Environment cleanup needed

   Evaluation in the TensorFlow environment emitted `No module named 'astropy'` from the dataset resolver path. The raw config fallback worked here, but the evaluation environment should include `astropy` or avoid resolver import dependency for standardized manifest configs.

## 8. Recommendation

Use this run as a successful first standardized training smoke test, not as the final model.

Recommended checkpoint for review:

`pv_shells/runs/pv_unet_standardized_5kpc_200kms_20260521_015917/best_model.keras`

Recommended operating thresholds:

- `0.05` for maximum-recall visual review.
- `0.075` for balanced high-recall candidate harvesting.
- `0.1` as a cleaner comparison layer.

Recommended next diagnostics before deciding on another run:

1. Generate review panels from `best_model.keras` at thresholds 0.05 and 0.075.
2. Run category-wise evaluation by manifest `cut_category`.
3. Inspect false positives among fine-grid and background cuts.
4. Inspect the 22 validation positive patch misses and 2 test positive patch misses from the best checkpoint.
5. Run deterministic connected-component candidate extraction on probability maps, preserving raw probability maps.
6. After standardized review, mine fresh hard negatives in the standardized coordinate system only.

## 9. Bottom Line

The standardized pipeline ran and produced a usable high-recall localizer checkpoint. `best_model.keras` is much more practical than `high_recall_model.keras`: it preserves nearly saturated patch recall while avoiding the all-negative-patch false-positive behavior of the high-recall checkpoint.

I would not launch a larger follow-up training run until category-wise deployment-style evaluation and visual review panels are generated. If those look acceptable, the next training run should focus on improving cross-galaxy generalization and review-load control without sacrificing the 0.05 / 0.075 recall regime.
