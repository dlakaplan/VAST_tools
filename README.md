# VAST_tools

## stack_images.py

Produce a stacked image of all epochs for a VAST field, optionally applying positional and flux corrections. Run with `--help` for more details.

Requires [SWarp](https://www.astromatic.net/software/swarp) which is also available via [conda-forge](https://anaconda.org/conda-forge/astromatic-swarp). Or, on `mortimer`:

```bash
module use /sharedapps/LS/cgca/modulefiles/
module load swarp
```

### Example

The following will produce a stacked image of the VAST_0012+00A field, assuming that the images are stored following the VAST data release structure.

A spreadsheet of the VAST pilot corrections in Excel format is required to apply the positional and flux corrections to each image before stacking. This can be obtained by download a copy of the corrections Google Sheet in Excel format, see the [VAST Wiki](https://github.com/askap-vast/vast-project/wiki/Pilot-Survey-Status-&-Data#catalog-corrections) for details.

```bash
#!/bin/bash

module use /sharedapps/LS/cgca/modulefiles/
module load swarp
conda activate vast-tools

python stack_images.py \
    --out output_dir \
    --clean \
    --type=tiles \
    --imagepath="/raid-17/LS/kaplan/VAST/EPOCH*/TILES/STOKESI_IMAGES/" \
    --qc "VAST Pilot Corrections.xlsx" \
    --beam \
    "0012+00A"
```
