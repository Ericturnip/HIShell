# Standardized Run Diagnostics

Checkpoint: `best_model.keras`

No training was launched during this diagnostics pass.

## Patch-Sampled Metrics

| threshold | patch precision | patch recall | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.8026 | 0.9972 | 8628 | 2122 | 24 |
| 0.075 | 0.8064 | 0.9972 | 8628 | 2072 | 24 |

## Inspection Targets

- Fine-grid false-positive patch samples at 0.075: 410
- Background false-positive patch samples at 0.075: 474
- Missed positive patch samples at 0.075: 24
- Missed positives by split: {'test': 2, 'val': 22}

Because the standardized config uses `samples_per_pv: 2`, these are patch-sample counts. Unique PV-cut counts are:

- Fine-grid false-positive PV cuts at 0.075: 205 unique cuts (130 validation, 75 test).
- Background false-positive PV cuts at 0.075: 237 unique cuts (126 validation, 111 test).
- Missed positive PV cuts at 0.075: 12 unique cuts (11 validation, 1 test).

The missed positives are concentrated in `ngc_4449`, `ngc_4736`, and one `ngc_3184` test cut. The only test miss is `ngc_3184__std_000273.npy`, a `spatial_offset_8pix__shell_minor` cut.

Detailed CSVs:

- `fine_grid_false_positives_0p075.csv`
- `background_false_positives_0p075.csv`
- `missed_positive_patches_0p075.csv`
- `category_metrics_by_cut_category.csv`
- `category_metrics_by_cut_category_aggregate.csv`

## Deployment-Like Categories at 0.075

| split | galaxy | cut category | patches | positives | negatives | patch precision | patch recall | patch FP | patch FN |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| test | ddo53 | background_random_negative | 24 | 0 | 24 | 0.0000 | 0.0000 | 6 | 0 |
| test | ddo53 | fine_grid_deployment_like | 190 | 2 | 188 | 0.0909 | 1.0000 | 20 | 0 |
| test | ho_ii | background_random_negative | 320 | 36 | 284 | 0.2812 | 1.0000 | 92 | 0 |
| test | ho_ii | fine_grid_deployment_like | 190 | 34 | 156 | 0.4359 | 1.0000 | 44 | 0 |
| test | ngc_3184 | background_random_negative | 320 | 6 | 314 | 0.0882 | 1.0000 | 62 | 0 |
| test | ngc_3184 | fine_grid_deployment_like | 190 | 10 | 180 | 0.2174 | 1.0000 | 36 | 0 |
| test | ngc_7793 | background_random_negative | 216 | 8 | 208 | 0.1143 | 1.0000 | 62 | 0 |
| test | ngc_7793 | fine_grid_deployment_like | 190 | 22 | 168 | 0.3056 | 1.0000 | 50 | 0 |
| val | ho_i | background_random_negative | 48 | 0 | 48 | 0.0000 | 0.0000 | 16 | 0 |
| val | ho_i | fine_grid_deployment_like | 190 | 4 | 186 | 0.0714 | 1.0000 | 52 | 0 |
| val | ngc_2366 | background_random_negative | 208 | 10 | 198 | 0.1562 | 1.0000 | 54 | 0 |
| val | ngc_2366 | fine_grid_deployment_like | 190 | 12 | 178 | 0.2143 | 1.0000 | 44 | 0 |
| val | ngc_4449 | background_random_negative | 160 | 12 | 148 | 0.0984 | 1.0000 | 110 | 0 |
| val | ngc_4449 | fine_grid_deployment_like | 190 | 20 | 170 | 0.1471 | 1.0000 | 116 | 0 |
| val | ngc_4736 | background_random_negative | 192 | 2 | 190 | 0.0270 | 1.0000 | 72 | 0 |
| val | ngc_4736 | fine_grid_deployment_like | 190 | 8 | 182 | 0.1429 | 1.0000 | 48 | 0 |

## Component Extraction

Full-PV probability maps/components were generated for: {'val': 2660, 'test': 3680}.

Outputs:

- `probability_maps/best_model/{split}/`
- `candidate_components_threshold_0p05.csv` / `.json`
- `candidate_components_threshold_0p075.csv` / `.json`

## Fresh Standardized Hard Negatives

Fresh train-fold hard-negative candidates: 500

Outputs:

- `hard_negative_candidates_standardized_train.json`
- `hard_negative_candidates_standardized_train.csv`

These candidates are review-only and were not added to any training manifest or config. They were mined from the standardized training fold only, so the held-out test fold remains isolated.

## Review Panels

Review PNGs were written to `review_panels_standardized/`, with index `review_panels_standardized_index.csv`.
