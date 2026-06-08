# Model vs Original Label Examples

Each type has two held-out examples: one with strong pixel-level overlap, and one where the patch is detected but the individual predicted pixels are messy.

Contours: catalog label is colored by shell type; prediction contour is red at threshold 0.075. The probability panel shows the raw U-Net output.

Use `all_type_examples_grid.png` for one compact overview, or the individual PNG/PDF files for larger slide panels.

Selected examples are listed in `selected_examples.csv`; all scored candidates are in `selected_example_scoring_table.csv`.
