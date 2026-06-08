# Scientific Figure Captions

These figures are styled like compact paper figures. Each is exported as PNG and PDF.

**Figure 1. PV examples and catalog masks.** Three standardized position-velocity cuts used as U-Net inputs. Grayscale shows HI intensity after robust display scaling; colored contours show catalog-derived masks for shell types 1, 2, and 3.

**Figure 2. Dataset and label composition.** Panel (a) shows positive and negative PV cuts in each split. Panel (b) shows labeled shell-object counts by catalog type.

**Figure 3. Training history.** Panel (a) shows train and validation loss for the clean physical baseline. Panel (b) shows validation patch precision and recall at the working threshold of 0.075.

**Figure 4. Post-processing metrics.** Patch-level precision and recall before filtering and after deterministic connected-component filters. On the test split, beam-area filtering gives precision = 0.826 and recall = 0.999. On the NGC 3031 stress split, beam-area filtering gives precision = 0.753 and recall = 0.976.

**Figure 5. Candidate component feature distributions.** Distributions of connected-component morphology and probability features for test-set candidates that survive the final filter. Black curves overlap a catalog label; red curves are candidate-only components.

**Figure 6. Probability-mass ranking.** Precision-recall tradeoff when candidates are ranked by integrated probability mass and only the top N are retained.

**Figure 7. Beam and stress diagnostics.** Panel (a) shows synthesized beam areas in standardized model pixels. Panel (b) compares edge-touching components across validation, test, and the NGC 3031 stress split.

## Files

- `fig01_pv_examples.png` / `fig01_pv_examples.pdf`
- `fig02_dataset_type_distribution.png` / `fig02_dataset_type_distribution.pdf`
- `fig03_training_history.png` / `fig03_training_history.pdf`
- `fig04_postprocessing_precision_recall.png` / `fig04_postprocessing_precision_recall.pdf`
- `fig05_candidate_feature_distributions.png` / `fig05_candidate_feature_distributions.pdf`
- `fig06_probability_mass_ranking.png` / `fig06_probability_mass_ranking.pdf`
- `fig07_beam_and_stress_diagnostics.png` / `fig07_beam_and_stress_diagnostics.pdf`
