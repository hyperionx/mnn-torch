# Publication reproduction notebooks

The four topic notebooks are the figure-generation source. They display all
figures inline and do not write image files unless `MNN_SAVE_FIGURES=1` is set.
`REPRODUCE.ipynb` executes them in canonical order and validates the 17-figure
contract, including the eight active manuscript figures.

Every topic notebook exposes the same controls:

```python
RUN_PROFILE = "reduced"       # reduced | publication | smoke
DEVICE = "auto"               # CUDA, then MPS, then CPU
WORKERS = "auto"
SAVE_FIGURES = False
OUTPUT_DIR = ...
OVERWRITE = False
RUN_EXTERNAL_DATA = False
ALLOW_DATA_DOWNLOADS = False
```

- `01_device_and_static.ipynb` covers measured device panels, authored
  architecture, schematics, and static-network results.
- `02_homeostasis.ipynb` runs paired reduced `MSNN`/`TemporalMCSNN`
  architecture-dependence experiments, then covers the liability/asset
  mechanism, fault-confusion, and reduced feature-map diagnostics.
- `03_temporal_memory.ipynb` generates temporal architectures and runs genuine
  reduced retention, store-recall, and four-dataset capability experiments by
  default. Set `MNN_USE_TEMPORAL_ARCHIVE=1` with the `publication` profile to
  replay a provenance-complete full-sweep archive; without one, publication
  mode displays the read-only manuscript targets.
- `04_representations.ipynb` runs genuine reduced N-MNIST `TemporalMCSNN` and
  MNE EEG `MCSNN` experiments by default and extracts their conv2 spikes live.
  It also documents explicit full-dataset downloads and strict optional
  long-run archive replay.

The `publication` profile redraws measured records, validated sample archives,
and authored assets without automatically launching expensive training. `smoke`
is offline and minimal. Summary-only aggregates are not accepted as figure data.
References under `assets/references/` are read-only visual targets, never
regenerated evidence.

The committed temporal fixture contains deterministic class-balanced source
samples from N-MNIST, SHD, and DVS Gesture; seq-MNIST reuses the genuine MNIST
fixture. Reduced runs use a nonstandard small repartition and three seeds, so
their scores and broad confidence intervals are explicitly labelled reduced
validation. They support mechanism and trend checks, not numerical replacement
of the 20-seed manuscript sweep.

The representation notebook needs no download for reduced or smoke execution.
Its fixture contains 60 genuine N-MNIST samples per class framed into ten bins
and 64 genuine MNE auditory/visual EEG epochs per class. Set both
`MNN_ALLOW_DATA_DOWNLOADS=1` and `MNN_DOWNLOAD_REPRESENTATION_DATA=1` only to
obtain the full upstream sources. Saving is refused for reference-only and
placeholder figures; manuscript directories and existing targets additionally
require `MNN_OVERWRITE=1`.
