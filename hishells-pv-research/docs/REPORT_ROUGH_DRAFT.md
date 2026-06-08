# A High-Recall Position-Velocity Search for HI Shells in THINGS Galaxies

## Draft Note

This is a rough paper draft for reference while writing the final report. It is not meant to be submitted unchanged. Add the final GitHub URL, choose the figures you actually want in the report, and adjust the wording so it sounds like you.

## Abstract

Neutral hydrogen shells trace how stellar feedback and gas motion reshape the interstellar medium in nearby galaxies. Existing shell catalogs are usually built by visual inspection of HI data cubes and position-velocity diagrams, which is slow and subjective. In this project I tested whether a U-Net segmentation model could act as a high-recall detector for HI shells in standardized position-velocity cuts from THINGS galaxies. I used the Bagetakos et al. shell catalog as the label source, generated physically standardized PV cuts with a fixed 5 kpc spatial width and 200 km/s velocity window, and trained the model to prioritize missing as few shell-containing patches as possible. At a probability threshold of 0.075, the model reached patch-level recall of 0.9996 on the held-out test split, with patch precision of 0.815. Pixel-level precision was lower, which means the model often knew a shell was present but did not always outline the exact catalog mask cleanly. A connected-component post-processing step improved test-set patch precision to 0.826 while keeping recall at 0.999. A physical deployment-grid test on DDO 53 recovered all three catalog shells at shell level, but the blind-grid patch precision was only 0.067. The main result is that the model is useful as a first-pass candidate finder, not as a finished autonomous shell catalog. The remaining problem is review-load control: reducing false positives without losing the high recall that makes the detector useful.

## 1. Scientific Motivation

HI shells and holes are low-density cavities in neutral atomic hydrogen. They can be produced when young massive stars ionize gas, drive winds, and explode as supernovae. They can also be affected by galaxy rotation, turbulence, tidal structure, and line-of-sight confusion. Because neutral hydrogen extends far beyond the optical disks of many galaxies, HI shells are useful for studying how feedback couples to the larger interstellar medium.

The basic observational problem is that shells are not always obvious in a single image. In a moment map, a shell may look like an elliptical hole or a partial rim. In a position-velocity diagram, the same object may show one side of an expanding shell, both sides of an expanding shell, or a stalled cavity with no clear velocity splitting. Brinks and Bajaja (1986) introduced the common type scheme: type 1 shells have both sides stalled, type 2 shells show one expanding side, and type 3 shells show both expanding sides in PV space. Bagetakos et al. (2011) later applied this kind of visual cataloging to THINGS galaxies.

My project asks a narrower question: can a machine-learning model find shell-containing PV cuts with very high recall? I cared more about missing few shells than about drawing perfect masks. That choice fits the practical use case. If the model is used as a discovery tool on a new galaxy, a false positive costs human review time. A false negative means the shell might never be inspected. For this project, I treated the U-Net as the sensitive first stage in a larger pipeline: U-Net probability map, connected components, beam-size filtering, velocity sanity checks, probability ranking, and then human review.

## 2. Data

The data come from THINGS, The HI Nearby Galaxy Survey, which observed nearby galaxies in the 21 cm neutral hydrogen line with the Very Large Array. I used THINGS FITS cubes for 19 galaxies available in my local data directory. IC 2574 was listed in the catalog setup but was skipped because the local cube was missing. The catalog labels came from the Bagetakos et al. (2011) HI hole tables, stored in the repository as `J_AJ_141_23_table2.dat.txt` and `J_AJ_141_23_table7.dat.txt`.

The available galaxies were split by whole galaxy instead of by random PV patch. This matters because nearby cuts from the same galaxy are not independent. The validation galaxies were Holmberg I, NGC 2366, NGC 4449, and NGC 4736. The test galaxies were DDO 53, Holmberg II, NGC 3184, and NGC 7793. NGC 3031 was held out as a stress test. The remaining galaxies were used for training. I treated NGC 3031 separately because the M81 group contains tidal gas and line-of-sight velocity confusion, so its PV diagrams contain long continuous structures that can look shell-like to a high-recall detector. This is not simply a bad FITS-file problem. It is also a real astrophysical confusion problem.

The raw FITS cubes are too large for the GitHub repository, so the repository contains configs, source code, figures, result summaries, and the small catalog text files. The expected raw-data layout is documented in `docs/DATA.md`. The GitHub repository also contains a hook that blocks raw FITS cubes, generated numpy arrays, model weights, and TensorBoard logs from being committed.

## 3. Methods

### 3.1 Standardized PV cuts

The first step was to make all model inputs comparable in physical units. Native FITS cubes differ in spatial scale, spectral step, and sometimes header conventions. The module `src.pv.standardized_cuts` handles this part of the pipeline. A `StandardCutSpec` describes the galaxy, cut center, position angle, and physical window. The function `sample_standardized_pv` samples the cube into a fixed model image with shape `(96, 256)`.

Each PV cut spans 5 kpc across the spatial axis and 200 km/s along the velocity axis. The spatial axis is interpolated so the full 5 kpc window always maps to 256 columns. The velocity axis is centered on the local intensity-weighted moment-1 velocity, then interpolated into 96 rows. I also corrected blank-unit or FELO-like spectral axes into km/s with `physical_velocity_axis_kms`.

One detail that mattered was edge behavior. If a requested velocity window would extend below zero in the native cube coordinate frame, the code shifts the window upward so the final window is still 200 km/s wide. It does not truncate the array. Missing data at cube boundaries are padded with a local valid median rather than artificial Gaussian noise. The goal was to keep the model input shape fixed without teaching the network that boundary pixels contain fake random structure.

### 3.2 Labels and cut categories

The script `scripts.prepare_standardized_training_data.py` generated the clean physical baseline dataset. It created several cut categories:

- catalog-centered cuts over known shell centers
- spatial-offset and grazing cuts near catalog shells
- velocity-offset cuts to test sensitivity to velocity centering
- fine-grid cuts that mimic blind deployment
- background cuts from regions away from catalog shells

The catalog-derived label masks were generated in PV space and then cleaned with `_erase_isolated_subbeam_components`. This step removed isolated positive mask components smaller than the synthesized beam while preserving boundary-grazing shells. The beam information came from FITS header values such as `BMAJ`, `BMIN`, and pixel scale when available. This is important because a positive label smaller than the telescope beam is not physically resolvable. Keeping such labels would train the model to fit noise-scale marks.

For this clean baseline run I removed old hard-negative mining from the training data. That decision made the run easier to interpret. If performance changed, I wanted it to be caused by the physical standardization and the high-recall training setup, not by a mixture of old mined negatives from a different preprocessing pipeline.

### 3.3 U-Net model and loss

I trained a U-Net-like convolutional neural network for binary segmentation of PV cuts. A U-Net is useful here because it combines local image features with larger-scale context. The network output is a probability map with the same `(96, 256)` shape as the input.

The training loss used a blend of binary cross-entropy and Tversky loss, implemented in `src.train.losses.BCETverskyLoss`. The Tversky term is useful when false negatives and false positives should not be weighted equally. I used alpha = 0.3 for false positives and beta = 0.7 for false negatives, so missed shell pixels were penalized more strongly than extra predicted pixels. This matched the goal of building a high-recall detector.

I tracked both pixel-level and patch-level metrics. Pixel metrics ask whether the model drew the same mask as the catalog. Patch metrics ask whether the model put any probability above threshold in a patch that contains a shell. For this project, patch recall was the main scientific metric because I wanted to know whether a shell-containing cut would be sent to a reviewer.

### 3.4 Post-processing and deployment grid

After inference, I thresholded raw U-Net probability maps at 0.075 and measured connected components with `scripts.postprocess_component_filter_eval.py`. For each component I measured area, beam area, area divided by beam area, velocity extent, spatial extent, maximum probability, mean probability, integrated probability mass, bounding-box filling fraction, and edge-touching flags.

The main post-processing rule was a beam-area filter: components smaller than 1.2 times the beam area were removed unless they were catalog-associated or grazing examples in evaluation mode. Velocity extent and velocity-edge flags were treated more carefully. A component that touches the top or bottom velocity edge may be a large-scale kinematic structure rather than a shell, especially in NGC 3031. However, making this filter too aggressive can remove true positives. For that reason I treated velocity extent first as a diagnostic and used probability mass as a ranking score for human review.

To test a more realistic deployment mode, I also generated a physical grid for DDO 53. The grid was aligned with the galaxy major and minor axes and used a 0.09 kpc stride. This test asks a harder question than catalog-centered evaluation: if I cut a new galaxy blindly on a fine physical grid, do the catalog shells appear in at least one cut, and does the model flag them?

## 4. Results

### 4.1 Training behavior

The training loss decreased from 0.369 at epoch 1 to 0.243 at epoch 21. The validation loss moved around rather than decreasing smoothly, which is expected for a small held-out set of whole galaxies. Figure 3 shows the training history. The validation patch recall at threshold 0.075 stayed high throughout training, while pixel metrics were less stable. This already suggested that the model was better at detecting shell presence than at drawing exact catalog boundaries.

### 4.2 Held-out validation and test performance

At the working threshold of 0.075, the held-out test split had patch precision 0.815 and patch recall 0.9996. Only 2 positive test patches were missed out of 5084 positive patches. The validation split showed a similar pattern: patch precision 0.778 and patch recall 0.998.

Pixel metrics were lower. On the test split, pixel precision was 0.268 and pixel recall was 0.690. This is not a contradiction. It means the model often detected that a shell was present but did not always match the exact catalog mask. Figure 1 shows examples of catalog masks, and the model-label example panels in `docs/model_label_examples/` show the same issue more directly: some patches are clear patch-level hits but messy pixel-level segmentations.

The combined confusion matrices for validation, test, and stress splits are shown in Figures 8 and 9. At the patch level, the combined counts were 20,520 true positives, 6,148 false positives, 294 false negatives, and 3,014 true negatives. At the pixel level, the class imbalance is much larger because almost all pixels are background. This is why patch-level and pixel-level results should not be interpreted as the same task.

### 4.3 Stress test on NGC 3031

NGC 3031 was the hardest case. At threshold 0.075 it reached patch recall 0.977, but patch precision was 0.749. Pixel precision was only 0.032 and pixel recall was 0.282. The low pixel precision is consistent with the visual problem: NGC 3031 has long tidal or line-of-sight structures in PV space, so the network fires on extended features that are not cataloged shells.

This stress split was useful because it showed the limit of the detector. A high-recall U-Net can find shell-like signal, but it does not know by itself whether a continuous velocity feature is an expanding shell, tidal debris, or another large-scale kinematic structure. Figure 7 shows that NGC 3031 produced many edge-touching components. That is a useful diagnostic for future filtering, but it should be used carefully because a hard edge filter reduced stress-split recall from 0.976 after beam filtering to 0.949.

### 4.4 Component filtering

The connected-component layer improved precision slightly without destroying recall. On the test split, the raw patch-level metrics were precision 0.815 and recall 0.9996. After beam-area filtering, precision rose to 0.826 and recall stayed at 0.999. Adding a velocity-extent filter gave precision 0.827 with the same recall. The improvement is modest, but the important result is that the deterministic filter did not wipe out real detections.

On NGC 3031, the beam filter improved patch precision from 0.749 to 0.753 while recall stayed at 0.976. A velocity-edge filter removed more false positives, but it also removed more true positives. That result supports a conservative post-processing strategy: use beam size as a hard filter, use velocity-edge behavior as a flag, and rank candidates by integrated probability mass for review.

### 4.5 Blind physical-grid test

The DDO 53 physical-grid test used 20,000 fine-grid cuts and recovered all three catalog shells at shell level. Shell-level grid coverage was 1.0, and shell-level model detection recall was also 1.0. This was encouraging because it showed that a fine physical grid can cover the known shells without relying on catalog-centered cuts.

The patch-level grid result was less clean: precision was 0.067 and recall was 0.941. That means the deployment mode would still produce many false candidate patches. I do not see that as a failure of the model. It is the cost of using a sensitive detector on a blind grid. The next stage needs better review-load control, not necessarily immediate retraining.

## 5. Discussion and Uncertainty

The main result is that the model can work as a high-recall shell finder in standardized PV space. It is not accurate enough to replace a human cataloger. The useful output is a ranked candidate list and a set of probability maps that help a human decide where to look.

The main uncertainty comes from the labels and the validation design. The Bagetakos catalog is based on visual identification, so the labels are not perfect ground truth. Some model "false positives" may be real shell-like structures that were not cataloged. Other false positives are probably ordinary HI turbulence, velocity crowding, or tidal structure. This is especially true in NGC 3031.

The galaxy-held-out split is a better test than random patch splitting, but the sample is still small. The validation and test splits contain only four galaxies each, and the physical deployment grid was tested in detail on DDO 53. A stronger result would repeat the physical-grid test across every held-out galaxy and compare shell-level recall and candidate load by galaxy type, inclination, and beam size.

There is also a difference between detecting a patch and measuring a shell. Patch recall answers the first question: did the model notice something shell-like? Pixel precision and pixel recall answer a harder question: did it trace the catalog mask? For a final catalog, I would need better boundary fitting, perhaps by using component-level ellipses or by fitting shell geometry after the U-Net step.

## 6. Next Steps

The next step is to turn the high-recall detector into a practical review tool. I would keep the U-Net threshold low, then improve post-processing. Beam-size filtering should remain a hard cut because it is tied to the telescope resolution. Velocity extent and velocity-edge contact should probably become ranked warning features rather than immediate rejection rules. Probability mass, area over beam, and compactness can be used to sort candidates so that a human reviews the strongest ones first.

I would also expand the physical-grid evaluation. The DDO 53 result shows that a fine grid can recover known shells, but one galaxy is not enough. A better deployment test would generate major/minor-axis grids for every held-out test galaxy and report shell-level recall, patch precision, and candidate count per square kiloparsec.

A second improvement would be better classification of shell type. The current model predicts a binary shell mask. The Bagetakos/Brinks type labels contain more physical information: type 1 for stalled shells, type 2 for one-sided expansion, and type 3 for two-sided expansion. A follow-up model could first detect shell candidates and then classify candidate components by PV morphology. This would make the review list easier to use.

Finally, the project should include more robust uncertainty estimates. I would retrain the model with different random seeds, test threshold sensitivity more systematically, and compare the U-Net against simpler baselines such as matched filters or classical connected-component methods. This would make it clearer how much of the performance comes from the neural network and how much comes from the standardized PV representation.

## 7. Software and Reproducibility

All project code is organized in a GitHub-ready repository:

```text
TODO: insert GitHub URL here
```

The main implementation files are documented in `docs/METHODS_CODE_MAP.md`. The figure-generation path is documented in `docs/FIGURE_REPRODUCIBILITY.md`. The most important scripts are:

- `scripts.prepare_standardized_training_data.py` for dataset generation
- `src.pv.standardized_cuts.py` for physically standardized PV cuts
- `src.train.losses.py` for BCE plus Tversky loss
- `src.train.callbacks.py` for segmented validation metrics
- `scripts.evaluate_pv_unet.py` for model evaluation
- `scripts.postprocess_component_filter_eval.py` for connected-component filtering
- `docs/figures/generate_scientific_figures.py` for Figures 1-7
- `docs/figures/generate_confusion_matrices.py` for Figures 8-11

I verified the code by inspecting data manifests, checking PV/label plots, comparing saved metric files, rerunning figure scripts, and running syntax checks with:

```bash
python -m compileall -q src scripts docs/figures tests
```

One full `pytest -q` run did not complete because the local Python 3.13 environment segfaulted while importing `pyarrow` through pandas/Keras. I would not hide that in the final report. I would say that syntax checks passed, figures and result files were regenerated from scripts, and the remaining test issue appears to be an environment binary problem.

## 8. References

Bagetakos, I., Brinks, E., Walter, F., de Blok, W. J. G., Usero, A., Leroy, A. K., Rich, J. W., & Kennicutt, R. C. 2011, "The Fine-Scale Structure of the Neutral Interstellar Medium in Nearby Galaxies," The Astronomical Journal, 141, 23. DOI: 10.1088/0004-6256/141/1/23.

Brinks, E., & Bajaja, E. 1986, "A high resolution hydrogen-line survey of Messier 31. III: H I holes in the interstellar medium," Astronomy & Astrophysics, 169, 14-42.

Walter, F., Brinks, E., de Blok, W. J. G., Bigiel, F., Kennicutt, R. C., Thornley, M. D., & Leroy, A. 2008, "THINGS: The HI Nearby Galaxy Survey," The Astronomical Journal, 136, 2563-2647. DOI: 10.1088/0004-6256/136/6/2563.
