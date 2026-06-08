# Next Training Setup Report

Run context: `pv_unet_overnight_20260520_015732`
Previous model: `best_model.keras`
Setup date: 2026-05-20

No training was launched during this setup pass.

## 1. Standardized PV Generation Status

Status: ready for the next training run, with one caveat: beam metadata is unavailable for these cube headers/configs, so beam-based offsets fall back to documented pixel offsets.

Implemented/verified:

- fixed spatial window: `spatial_window_kpc = 5.0`
- fixed velocity window: `velocity_window_kms = 200.0`
- velocity range: local Moment-1 velocity +/- 100 km/s
- velocity axis correction for THINGS FELO/VELO headers with blank/non-km units that behave like m/s
- metadata per cut includes source cube path, Moment-1 path, adopted distance, angular window, pixel window, pixel scale, local velocity, velocity min/max, channel indices, PV center pixel, PV center RA/Dec, angle, category, shell ID/offsets where applicable, and quality flags

Core files:

- `src/pv/standardized_cuts.py`
- `scripts/prepare_standardized_training_data.py`

Fallback behavior:

- If Moment-1 along the cut is unavailable, the code falls back to the resolved systemic velocity.
- If beam metadata is missing, spatial offset categories use `4`, `8`, and `16` pixel offsets and flag `beam_missing_used_pixel_offsets`.

## 2. Dataset Regeneration Status

Standardized dataset created at:

`pv_shells/training_data/standardized_5kpc_200kms/`

Manifest/config outputs:

- `train_manifest.csv`
- `val_manifest.csv`
- `test_manifest.csv`
- `splits/train_manifest.txt`
- `splits/val_manifest.txt`
- `splits/test_manifest.txt`
- `train_standardized_high_recall.yaml`
- `standardized_split_summary.json`
- `finite_fraction_audit.json`

The `.txt` manifests are for the current `tf.data` loader; the CSV manifests carry the richer metadata requested for audit/review.

Split summary:

| split | samples | positives | negatives | galaxies |
|---|---:|---:|---:|---:|
| train | 23,784 | 17,400 | 6,384 | 11 |
| val | 2,660 | 1,779 | 881 | 4 |
| test | 3,680 | 2,547 | 1,133 | 4 |

Finite-data audit:

| split | low finite fraction `<0.25` | minimum finite fraction |
|---|---:|---:|
| train | 0 | 0.516 |
| val | 0 | 1.000 |
| test | 0 | 1.000 |

Dataset self-test passed in the TensorFlow environment:

- batch shape: `(8, 96, 256, 1)`
- labels include both `0` and `1`

## 3. Validation Realism

The new validation set is no longer only centered catalog cuts. It includes centered positives, spatial offsets, angle offsets, velocity offsets, grazing/nearby cuts, fine-grid deployment-like cuts, and background negatives.

Validation category summary:

| validation cut_category | count | positives | negatives |
|---|---:|---:|---:|
| angle_offset_+22.5deg | 76 | 76 | 0 |
| angle_offset_+45deg | 76 | 76 | 0 |
| angle_offset_-22.5deg | 76 | 76 | 0 |
| angle_offset_-45deg | 76 | 76 | 0 |
| background_random_negative | 304 | 12 | 292 |
| centered_positive__galaxy_major | 76 | 76 | 0 |
| centered_positive__galaxy_minor | 76 | 76 | 0 |
| centered_positive__shell_major | 76 | 76 | 0 |
| centered_positive__shell_minor | 76 | 76 | 0 |
| fine_grid_deployment_like | 380 | 22 | 358 |
| random_nearby_grazing | 152 | 152 | 0 |
| spatial_offset_16pix__shell_major | 152 | 80 | 72 |
| spatial_offset_16pix__shell_minor | 152 | 92 | 60 |
| spatial_offset_4pix__shell_major | 152 | 137 | 15 |
| spatial_offset_4pix__shell_minor | 152 | 145 | 7 |
| spatial_offset_8pix__shell_major | 152 | 106 | 46 |
| spatial_offset_8pix__shell_minor | 152 | 122 | 30 |
| velocity_offset_+15kms | 76 | 76 | 0 |
| velocity_offset_+30kms | 76 | 75 | 1 |
| velocity_offset_-15kms | 76 | 76 | 0 |
| velocity_offset_-30kms | 76 | 76 | 0 |

Per-galaxy validation deployment-like counts:

| split | galaxy | cut_category | count | positives | negatives |
|---|---|---|---:|---:|---:|
| val | ho_i | fine_grid_deployment_like | 95 | 2 | 93 |
| val | ngc_2366 | fine_grid_deployment_like | 95 | 6 | 89 |
| val | ngc_4449 | fine_grid_deployment_like | 95 | 10 | 85 |
| val | ngc_4736 | fine_grid_deployment_like | 95 | 4 | 91 |
| val | ho_i | random_nearby_grazing | 12 | 12 | 0 |
| val | ngc_2366 | random_nearby_grazing | 52 | 52 | 0 |
| val | ngc_4449 | random_nearby_grazing | 40 | 40 | 0 |
| val | ngc_4736 | random_nearby_grazing | 48 | 48 | 0 |
| val | ho_i | background_random_negative | 24 | 0 | 24 |
| val | ngc_2366 | background_random_negative | 104 | 5 | 99 |
| val | ngc_4449 | background_random_negative | 80 | 6 | 74 |
| val | ngc_4736 | background_random_negative | 96 | 1 | 95 |

Validation is now deployment-like enough to test blind-grid false positives and some blind-grid positives. It still contains centered cuts by design, so metrics should also be reported by category rather than only as one aggregate validation number.

## 4. Galaxy Split And Leakage Check

The current code supports galaxy-level splits. `prepare_standardized_training_data.py` writes train/val/test manifests by galaxy.

Train galaxies:

`ddo154`, `ngc_2403`, `ngc_2841`, `ngc_2976`, `ngc_3031`, `ngc_3521`, `ngc_3627`, `ngc_4214`, `ngc_5194`, `ngc_628`, `ngc_6946`

Validation galaxies:

`ho_i`, `ngc_2366`, `ngc_4449`, `ngc_4736`

Test galaxies:

`ddo53`, `ho_ii`, `ngc_3184`, `ngc_7793`

Leakage safeguards:

- split is by galaxy, not by random patch
- old hard negatives are not included
- hard negatives, if added later, must remain locked to their source-galaxy fold
- held-out test galaxies should not be mined into training unless a new clean holdout is defined

## 5. Beam-Aware Label Sanity Check

Implemented:

- `scripts/audit_label_components.py`

Output:

- `pv_shells/training_data/standardized_5kpc_200kms/label_component_audit.csv`
- `pv_shells/training_data/standardized_5kpc_200kms/label_component_audit.json`

Audit results:

| item | count |
|---|---:|
| masks audited | 30,124 |
| positive masks | 21,726 |
| positive components | 40,003 |
| resolvable_positive | 37,995 |
| grazing_positive | 1,711 |
| isolated_subbeam_speck | 0 |
| ambiguous_small_positive | 297 |
| labels changed | 0 |

Default cleaning mode:

```yaml
label_cleaning_mode: audit_only
```

No labels were erased or ignored. Grazing positives are preserved. The 297 ambiguous small positives should be reviewed visually before any future cleaning mode is enabled.

## 6. Loss And Config Readiness

Implemented/verified in `src/train/losses.py`:

- binary crossentropy
- weighted BCE
- Tversky loss
- BCE + Tversky
- weighted BCE + Tversky
- pixel precision/recall at `0.05`, `0.075`, `0.1`
- patch precision/recall at `0.05`, `0.075`, `0.1`

Prepared default loss in `train_standardized_high_recall.yaml`:

```yaml
loss:
  name: bce_tversky
  tversky_alpha: 0.3
  tversky_beta: 0.7
  bce_weight: 0.5
  tversky_weight: 0.5
```

Model checkpointing now saves:

- `best_model.keras`, monitored by `val_pr_auc`
- `high_recall_model.keras`, monitored by `val_patch_recall_0p075`

This keeps the old PR-AUC selection path while adding a recall-oriented checkpoint.

## 7. Hard Negatives

Old hard-negative file found:

`pv_shells/runs/pv_unet_overnight_20260520_015732/hard_negative_candidates.json`

This file is from the previous non-standardized coordinate representation. It was not included in:

- `train_manifest.csv`
- `val_manifest.csv`
- `test_manifest.csv`
- `train_standardized_high_recall.yaml`

Search confirmed no hard-negative references inside `training_data/standardized_5kpc_200kms/`.

Future plan: after training the standardized model, mine fresh hard negatives in standardized 5 kpc / 200 km/s local-velocity coordinates.

## 8. Post-Processing Readiness

Deterministic candidate extraction is prepared separately from training:

- `scripts/extract_candidate_components.py`
- shared component feature utilities in `src/eval/diagnostic_utils.py`

It preserves probability maps until threshold/component extraction and supports threshold tables at:

- `0.05`
- `0.075`

Features include component area, max/mean probability, integrated probability mass, centroid, bounding box, aspect ratio, distance from edge, edge-touching flag, velocity extent, spatial extent in kpc when metadata exists, and beam-area units when beam metadata exists.

Physics-informed behavior:

- tiny isolated components can be flagged
- grazing shell components are not automatically removed
- edge-touching components are flagged
- strange velocity extents are flagged, not hard-deleted

## 9. Final Recommendation

Setup is ready for a new standardized high-recall training run tonight.

Remaining caveats:

- Beam metadata is missing for all 19 generated galaxies, so spatial offsets use `4`, `8`, and `16` pixel fallbacks rather than true `0.5`, `1`, and `2` beam offsets.
- Validation is much more deployment-like than before, but final model selection should still be reported by cut category because centered catalog cuts remain present.
- Ambiguous small positive components were audited but not modified.

Exact command to run later, not executed here:

```bash
cd "/Users/turnip/Desktop/HIShells copy/pv_shells"
/opt/homebrew/anaconda3/envs/tf/bin/python -m src.train.train_pv_unet --config training_data/standardized_5kpc_200kms/train_standardized_high_recall.yaml --run pv_unet_standardized_5kpc_200kms_$(date +%Y%m%d_%H%M%S) --quiet --every 1
```
