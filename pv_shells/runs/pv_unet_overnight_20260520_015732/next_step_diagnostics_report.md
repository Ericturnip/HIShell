# PV U-Net Next-Step Diagnostics Report

Run: `pv_unet_overnight_20260520_015732`
Model: `best_model.keras`
Date: 2026-05-20

No new training was launched during this diagnostic phase.

## 1. Current Dataset And Split Audit

The current overnight config is `pv_shells/training_data/combined_train_overnight.yaml`, with `output_root` set to `pv_shells/training_data/combined`.

The split is by whole galaxy, not random PV patch:

| split | galaxies | patches | positive | negative |
|---|---:|---:|---:|---:|
| train | 11 | 18,596 | 10,712 | 7,884 |
| val | 4 | 2,056 | 1,003 | 1,053 |
| test | 4 | 2,872 | 1,495 | 1,377 |

Train galaxies: `ddo154`, `ngc_2403`, `ngc_2841`, `ngc_2976`, `ngc_3031`, `ngc_3521`, `ngc_3627`, `ngc_4214`, `ngc_5194`, `ngc_628`, `ngc_6946`.

Validation galaxies: `ho_i`, `ngc_2366`, `ngc_4449`, `ngc_4736`.

Test galaxies: `ddo53`, `ho_ii`, `ngc_3184`, `ngc_7793`.

The test split is galaxy-held-out, which is good, but it is still dominated by catalog-driven cuts:

| test PV source | count |
|---|---:|
| catalog_shell | 1,320 |
| background_negative | 1,320 |
| grid | 232 |

Catalog shell cuts include offset fractions `-0.5`, `0.0`, and `0.5`, with 440 cuts at each offset in test. There are 440 perfectly centered test catalog cuts. This is a generalization risk: held-out galaxies prevent direct galaxy leakage, but many positive validation/test examples are still idealized Bagetakos-centered or near-centered PV cuts. Blind deployment will require fine-grid cuts that often only partially intersect shells.

Patch metadata currently traces galaxy, shell ID for catalog cuts, pixel center, cut direction/orientation, spatial offset fraction, full native velocity axis range, PV source type, position samples, and labels. It does not reliably include RA/Dec per patch, local Moment-1 velocity, local velocity window, Moment-1 map source, or fixed physical spatial-window metadata.

Detailed audit artifact: `dataset_split_audit.json`.

## 2. Current PV Representation Audit

Current spatial sampling is not fixed in kpc:

- Grid cuts use arcsec extents and arcsec step from config, typically `x_extent_arcsec = y_extent_arcsec = 420`.
- Catalog-shell positive cuts use shell radius-scaled half-lengths with min/max pixel bounds.
- Training crops are fixed model patches, `patch_vel = 96`, `patch_pos = 256`, after padding/cropping.

Current velocity sampling is not local-disk centered:

- PV arrays retain the native cube spectral axis.
- Metadata records ranges such as `68960.98 -> -34114.77` with a step of `-2576.89`, even though THINGS `FELO-HEL` headers have blank `CUNIT3`; these values behave like m/s, not km/s.
- Labels use Bagetakos/catalog shell velocity centers. Inputs are not recentered around local Moment-1 velocity.

This creates cross-galaxy mismatch: shells can appear at different absolute velocity locations and with different physical spatial scales across galaxies.

## 3. Physical Standardization Prepared

Added a prepared standardized extraction path:

- `src/pv/standardized_cuts.py`
- `scripts/generate_standardized_pv_cuts.py`

Default prepared representation:

- fixed spatial window: 5 kpc
- fixed velocity window: 200 km/s total
- velocity range: local Moment-1 velocity ±100 km/s
- local velocity from an intensity-weighted Moment-1 map derived from the cube
- deployment-grid mode with multiple angles per grid point

The code records:

- galaxy
- adopted distance
- spatial window in kpc
- angular window in arcsec
- pixel window size
- pixel scale
- PV center and angle
- local velocity
- velocity min/max and channel indices
- Moment-1 source and fallback method

A dry-run manifest was generated for NGC 3184 at:

`standardized_cut_dry_run_ngc3184/standardized_cut_manifest.json`

The dry run also flags the current velocity-unit issue and corrects FELO-like blank-unit axes from m/s-like values to km/s-like values.

## 4. Threshold Recommendation

Existing test threshold metrics:

| threshold | pixel recall | pixel precision | patch recall | patch precision |
|---:|---:|---:|---:|---:|
| 0.05 | 0.777 | 0.262 | 0.9993 | 0.787 |
| 0.075 | 0.737 | 0.303 | 0.9987 | 0.804 |
| 0.1 | 0.705 | 0.337 | 0.9977 | 0.819 |

Recommendation remains:

- `0.05`: maximum recall harvesting mode.
- `0.075`: preferred balanced high-recall review mode.
- `0.1`: cleaner but more conservative.

Because the U-Net is being used as a high-recall localizer, not the final catalog generator, `0.05` and `0.075` should both remain active in review and candidate extraction.

## 5. Cut-Offset Robustness

New outputs:

- `cut_offset_robustness_eval.json`
- `cut_offset_robustness_eval.csv`

At threshold 0.075:

| category | n | patch recall | patch FP count | pixel recall | pixel precision |
|---|---:|---:|---:|---:|---:|
| centered catalog positives | 440 | 1.000 | 0 | about 0.72-0.78 by orientation | about 0.36-0.41 |
| 0.5-radius spatial-offset positives | 880 | 1.000 | 0 | about 0.71-0.78 by orientation | about 0.34-0.38 |
| fine-grid positives | 81 | 1.000 | 0 | 0.649 | 0.210 |
| random-background negatives | 1,226 | n/a | 256 | n/a | n/a |
| fine-grid negatives | 151 | n/a | 79 | n/a | n/a |

Interpretation:

- The model still detects shell-containing patches when cuts are centered or offset by the existing 0.5 projected-radius augmentation.
- Deployment-like grid positives are still detected at patch level, but masks are blobby and lower precision.
- Negative grid/background cuts create a significant review load at high-recall thresholds.

Synthetic spectral-offset tests show sensitivity in pixel alignment:

- `+30 km/s` shifted centered positives at 0.075: patch recall 1.000, pixel recall 0.142.
- `-30 km/s` shifted centered positives at 0.075: patch recall 1.000, pixel recall 0.027.

This supports local velocity centering before retraining. Patch-level detection survives, but pixel-level shell localization degrades badly when the velocity placement is wrong.

## 6. Visual Failure Analysis

New outputs:

- `review_panels/`
- `review_panels_index.csv`

Panels include:

- true positives at 0.075
- false positives at 0.075
- false positives at 0.05
- one 0.05-only detection
- one false negative at 0.075

Each panel shows PV input, label mask, raw probability map, masks at 0.05 and 0.075, and overlay. The stretch uses robust asinh-style scaling.

Preliminary failure patterns from the generated tables:

- false positives are common in random-background and fine-grid negative cuts
- candidate components often appear in grid/background PVs, consistent with high recall plus overplotting
- edge-touching components are non-negligible: 842 components at 0.05 and 569 at 0.075
- some “background negative” false positives may be uncataloged real structures or label incompleteness rather than purely model error

Manual review should inspect the highest-probability hard negatives before deciding whether they are true negatives.

## 7. Hard-Negative Mining

New output:

- `hard_negative_candidates.json`

The mining pass found 358 candidate hard negatives at thresholds 0.05 and/or 0.075:

- 275 from `background_negative`
- 83 from `grid`

Top false positives have max probabilities near 0.94-0.99 and mask areas of hundreds of pixels. These are useful review targets.

Recommendation:

- Do not add these automatically to training.
- First classify them as obvious artifacts, ambiguous real structures, or likely uncataloged shells.
- Keep each hard negative tied to its source galaxy fold.
- Do not use held-out test hard negatives in the next training set unless a new clean holdout is defined.

## 8. Cross-Galaxy Validation Roadmap

The current code can already split by galaxy via `prepare_combined_training_data.py`; the existing run used galaxy-held-out manifests.

Recommended next validation framework:

1. Keep a clean final holdout galaxy set untouched.
2. Add leave-one-galaxy-out or rotating-galaxy validation configs.
3. Report metrics separately per galaxy and per cut type: catalog centered, offset catalog, grid positive, grid negative, background negative.
4. Lock hard negatives to the same fold as their source galaxy.
5. If test false positives are used for training analysis, redefine a new final holdout before claiming final performance.

Missing or weak metadata for robust rotation:

- explicit RA/Dec per patch
- fixed physical window metadata
- local velocity metadata
- beam metadata for beam-unit offsets and component filters
- Moment-1 source path/hash

## 9. Post-Processing Roadmap

New outputs:

- `candidate_components_threshold_0p05.csv`
- `candidate_components_threshold_0p05.json`
- `candidate_components_threshold_0p075.csv`
- `candidate_components_threshold_0p075.json`

Extracted components:

| threshold | components |
|---:|---:|
| 0.05 | 5,565 |
| 0.075 | 4,838 |

The deterministic second layer should rank and filter components using:

- connected-component area
- max/mean/integrated probability
- bounding box and centroid
- aspect ratio and extents
- edge-touching flag
- velocity extent
- spatial extent
- beam-area filtering when beam metadata is available
- soft warnings for implausibly large velocity extents

This should remain deterministic and interpretable before considering a second learned model.

## 10. Training Decision

Do not retrain immediately.

Recommended next step is review and pipeline restructuring:

1. Review panels and hard negatives.
2. Decide whether high-probability “false positives” are artifacts or uncataloged shells.
3. Standardize PV extraction to fixed 5 kpc windows and local Moment-1-centered ±100 km/s velocity windows.
4. Generate a deployment-like validation subset with fine-grid centers and multiple angles.
5. Preserve a clean galaxy holdout before adding hard negatives.
6. Then run the next training experiment with physical standardization and fold-safe hard negatives.

No new training was launched during this diagnostic phase.
