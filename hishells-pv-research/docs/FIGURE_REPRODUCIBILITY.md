# Figure Reproducibility

This file records how the report figures were generated. All paths are relative to the repository root.

## Commands

```bash
python docs/figures/generate_scientific_figures.py
python docs/figures/generate_confusion_matrices.py
python docs/figures/generate_model_label_examples.py --plot-only
```

Run `docs/figures/generate_model_label_examples.py` without `--plot-only` only when the saved model file is available locally, because that mode recomputes probability maps.

## Scientific Figures

- Figure 1, PV examples with catalog masks: `docs/figures/generate_scientific_figures.py`, `figure_pv_examples`.
- Figure 2, split and shell-type distributions: `docs/figures/generate_scientific_figures.py`, `figure_dataset_and_types`.
- Figure 3, training history: `docs/figures/generate_scientific_figures.py`, `figure_training_history`.
- Figure 4, post-processing precision and recall: `docs/figures/generate_scientific_figures.py`, `figure_postprocessing_metrics`.
- Figure 5, component feature distributions: `docs/figures/generate_scientific_figures.py`, `figure_candidate_features`.
- Figure 6, probability-mass ranking: `docs/figures/generate_scientific_figures.py`, `figure_probability_ranking`.
- Figure 7, beam and stress diagnostics: `docs/figures/generate_scientific_figures.py`, `figure_beam_and_stress`.

## Confusion Matrices

- Figure 8, patch-level confusion matrix: `docs/figures/generate_confusion_matrices.py`, `plot_confusion`.
- Figure 9, pixel-level confusion matrix: `docs/figures/generate_confusion_matrices.py`, `plot_confusion`.
- Figure 10, DDO 53 physical-grid patch matrix: `docs/figures/generate_confusion_matrices.py`, `plot_confusion`.
- Figure 11, DDO 53 physical-grid pixel matrix: `docs/figures/generate_confusion_matrices.py`, `plot_confusion`.

The helper `load_counts` combines saved evaluation counts across validation, test, and stress splits. The matrix layout is true-label rows by predicted-label columns.

## Model And Label Example Panels

- Compact six-example grid: `docs/figures/generate_model_label_examples.py`, `plot_grid`.
- Individual three-panel examples: `docs/figures/generate_model_label_examples.py`, `plot_example`.
- Example selection table: `docs/figures/generate_model_label_examples.py`, `score_examples`.
- Probability-map scoring: `docs/figures/generate_model_label_examples.py`, `score_examples_with_model`.

The saved tables in `docs/model_label_examples/` record which patches were selected and which probability maps were plotted.
