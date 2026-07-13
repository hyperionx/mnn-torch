# Publication reproduction notebooks

The four topic notebooks are the figure-generation source. They display all figures inline and do not write image files unless `MNN_SAVE_FIGURES=1` is set. `REPRODUCE.ipynb` executes them in canonical order and validates the 17-figure contract, including the eight active manuscript figures.

Every topic notebook exposes the same controls:

```python
RUN_PROFILE = "publication"   # reduced | publication | smoke
DEVICE = "auto"               # CUDA, then MPS, then CPU
WORKERS = "auto"
SAVE_FIGURES = False
OUTPUT_DIR = ...
OVERWRITE = False
RUN_EXTERNAL_DATA = False
ALLOW_DATA_DOWNLOADS = False
```

## Notebooks
- `01_device_and_static.ipynb` covers measured device panels, authored architecture schematics, and static-network results.
- `02_homeostasis.ipynb` runs paired `MSNN`/`TemporalMCSNN` architecture-dependence experiments, then covers the liability/asset mechanism, fault-confusion, and feature-map diagnostics.
- `03_temporal_memory.ipynb` generates temporal architectures and runs genuine retention, store-recall, and four-dataset capability experiments.
- `04_representations.ipynb` runs genuine N-MNIST `TemporalMCSNN` and MNE EEG `MCSNN` experiments by default and extracts their conv2 spikes live. It also documents explicit full-dataset downloads.

## Execution Profiles

The repository supports three primary profiles to balance rigor with accessibility:

### 1. `reduced` (Live Validation)
The **`reduced`** profile is the default active mode for running the networks live without requiring days of GPU time or heavy dataset downloads. It uses the bundled `.npz` fixtures under `data/fixtures/` (e.g., `temporal_reduced_datasets.npz`, `representations_reduced_datasets.npz`) which contain curated, deterministic, class-balanced source samples from N-MNIST, SHD, DVS Gesture, and MNE EEG.

**Key features of `reduced` runs:**
- **Fast & Genuine:** They execute genuine training and inference cycles through the actual model architectures.
- **Three Seeds:** They use a nonstandard small repartition and three random seeds (instead of 20).
- **Trend Checks:** Their scores and broad confidence intervals are explicitly labelled as "reduced validation." They serve as working proofs of the code, mechanisms, and trends, but do not numerically replace the 20-seed manuscript sweeps.
- **No Downloads:** The required fixtures are committed in the repository, so `reduced` runs entirely offline.

### 2. `publication` (Strict Reproducibility)
The **`publication`** profile strictly redraws measured records, validated sample archives, and authored assets without automatically launching expensive training. 
- For static records (`01_device_and_static`), it replays the committed `data/devicefixed_data.json` archive.
- For heavy 20-seed sweeps (`03_temporal_memory` and `04_representations`), it requires explicitly injecting full long-run archive replays (e.g., `MNN_USE_TEMPORAL_ARCHIVE=1`). 
- **Read-Only Fallback:** If the heavy full-sweep archives are not present (which are omitted from Git due to size), `publication` mode will gracefully fall back to displaying the read-only manuscript reference images from `assets/references/`. Summary-only `.npy` aggregates are explicitly forbidden and rejected as figure data to maintain provenance rigor.

### 3. `smoke` (CI Testing)
The `smoke` profile performs minimal, offline dummy execution passes to verify the code runs without crashing.

## Saving Figures
Saving is refused for reference-only and placeholder figures. To write the generated outputs to disk, set `MNN_SAVE_FIGURES=1`. Overwriting manuscript directories and existing targets additionally requires `MNN_OVERWRITE=1`.
