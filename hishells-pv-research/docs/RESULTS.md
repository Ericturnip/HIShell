# Clean Physical Baseline Results

The included summaries come from the clean physical baseline run:

```text
pv_unet_clean_physical_baseline_20260522_022751
```

The model was trained on standardized physical cuts. NGC 3031 was removed from training and used as the stress split. The main goal was patch-level recall, with deterministic post-processing used to reduce the review load after the U-Net probability map.

The DDO 53 physical galaxy-frame grid test used a 0.09 kpc stride and recovered all three catalog shells at shell level. The patch-level result for that grid was:

```text
precision: 0.06691253951527924
recall: 0.9407407407407408
f1: 0.12493851451057549
```

The low grid precision is expected for blind deployment because the model is intentionally sensitive. Connected-component filtering, beam-area culling, velocity-edge flags, and probability-mass ranking are the review-load controls.
