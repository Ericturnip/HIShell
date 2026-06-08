# Methods To Code Map

Use this file when writing the methods section of the report. It names the modules and functions that implement each part of the project pipeline.

## Physical PV Cut Generation

- `src.pv.standardized_cuts.StandardCutSpec` describes one requested PV cut before sampling.
- `src.pv.standardized_cuts.sample_standardized_pv` samples a cube into a fixed `(96, 256)` PV image.
- `src.pv.standardized_cuts.physical_velocity_axis_kms` converts native FITS spectral axes into km/s.
- `src.pv.standardized_cuts.moment1_velocity_map` estimates the local intensity-weighted velocity field.
- `src.pv.standardized_cuts.local_velocity_for_cut` chooses the velocity center for each PV cut.
- `src.pv.standardized_cuts._fixed_velocity_axis` keeps every cut at a full 200 km/s window.

## Clean Dataset Construction

- `scripts.prepare_standardized_training_data.main` runs the full clean physical baseline data generation.
- `scripts.prepare_standardized_training_data._catalog_specs` creates centered, offset, velocity-offset, and grazing cuts from catalog shells.
- `scripts.prepare_standardized_training_data._galaxy_frame_deployment_specs` creates fine-grid cuts in the galaxy major/minor-axis frame.
- `scripts.prepare_standardized_training_data._background_specs` creates random background negatives from real HI emission away from catalog shells.
- `scripts.prepare_standardized_training_data._write_cut` writes each PV array, label mask, metadata sidecar, and manifest row.
- `scripts.prepare_standardized_training_data._erase_isolated_subbeam_components` removes isolated sub-beam label speckles while preserving boundary-grazing shells.

## Training

- `src.pv.dataset.build_dataset` loads generated PV/label pairs as TensorFlow datasets.
- `src.train.losses.make_loss_and_metrics` builds the training loss and metrics from the clean-baseline config.
- `src.train.losses.BCETverskyLoss` blends BCE with Tversky loss.
- `src.train.losses.TverskyLoss` implements the false-negative-weighted Tversky term.
- `src.train.callbacks.SegmentedValidationCallback` logs validation metrics by cut category during training.

## Evaluation

- `scripts.evaluate_pv_unet.evaluate` computes pixel-level and patch-level metrics for a trained model.
- `src.train.callbacks.evaluate_model_by_category` reports the five segmented validation categories.
- `src.train.callbacks.canonical_cut_category` maps detailed cut names into report categories.

## Connected-Component Post-Processing

- `scripts.postprocess_component_filter_eval.evaluate_split` runs thresholding, component measurement, filtering, and ranking for one split.
- `scripts.postprocess_component_filter_eval._component_features` measures candidate area, beam area, extent, probability mass, and edge-touching flags.
- `scripts.postprocess_component_filter_eval._beam_area_pixels` chooses the best synthesized-beam estimate from FITS headers or metadata.
- `scripts.postprocess_component_filter_eval._rank_metrics` computes patch precision and recall among the top-N ranked candidates.

## Review Catalog

- `scripts.build_shell_review_catalog.main` assembles filtered candidates into a human-review table.
- `scripts.extract_candidate_components.py` and `scripts.export_review_panels.py` support candidate inspection panels.
