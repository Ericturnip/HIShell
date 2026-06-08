# Data Layout

The repo stores code, configs, small catalog tables, figures, and compact result summaries. It does not store THINGS FITS cubes or generated training arrays.

Use this local layout:

```bash
data/raw/
training_data/
runs/
```

`data/raw/` should contain the THINGS cube files named in `training_data/manifest.json`. `training_data/` and `runs/` are generated locally and ignored by Git, except for the checked-in config files.

NGC 3031 is held out as a stress-test galaxy. The main problem is not simply a bad FITS file. M81-group tidal structure produces continuous PV features that can look shell-like to a high-recall detector.
