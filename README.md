# mnn-torch

`mnn-torch` is a PyTorch reference implementation of device-grounded
memristive spiking neural networks. It accompanies *Working memory in spiking
neural networks from memristive device transients* and provides a lightweight,
auditable route from measured-device records or genuine per-sample experiment archives to the paper's
figures.

## Installation

Python 3.10 or newer is required. The package and each repository are
independent; no sibling checkout or shared Conda environment is needed. The
recommended `uv` setup resolves the official CUDA 13.0 PyTorch wheels on
Windows/Linux (and the CPU wheel on macOS):

```bash
uv sync --extra repro
.venv/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # Windows
```

The base install contains the model and data interfaces. The `repro` extra adds
Jupyter and plotting tools, `external` adds N-MNIST/embedding adapters, `test`
adds the test runner, and `release` adds package-build tools. For example:

```bash
uv sync --extra repro --extra external --extra test
```

For a manual Windows/Linux pip environment, install the CUDA build before this
project:

```bash
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[repro]"
```

`DEVICE="auto"` then selects CUDA first. An ordinary PyPI Windows Torch wheel
is CPU-only and will make `auto` fall back to CPU; verify
`torch.cuda.is_available()` before a reduced/publication run.

## Reproducing the figures

Start Jupyter in the repository root and run `experiments/REPRODUCE.ipynb`:

```bash
python -m jupyter lab experiments/REPRODUCE.ipynb
```

The topic notebooks form the readable reproduction record:

- `01_device_and_static.ipynb`: measured device behaviour, static networks,
  architecture, and reduced MSNN/MCSNN training.
- `02_homeostasis.ipynb`: live fault diagnostics and homeostatic mechanisms.
- `03_temporal_memory.ipynb`: temporal architectures plus live reduced
  retention, store-recall, and four-dataset capability experiments, with a
  strict optional full-sweep archive path.
- `04_representations.ipynb`: live reduced N-MNIST and EEG conv2 representation
  workflows, plus explicit source-download and optional archive paths.

Together they catalogue 17 first-party figures: eight used by the accompanying
manuscript, eight manuscript-adjacent figures, and two additional contextual
figures. Every figure record includes its data provenance and claim status.
Summary-only tables are never expanded into seed observations or plotted as
regenerated results. Quantitative outputs require measured records, genuine live
runs, or validated per-seed archives. Panels whose original runner or seed arrays
do not survive are displayed as read-only references and cannot be saved.

The default `reduced` static MNIST panels are generated live by the real `MSNN`
and `MCSNN` classes from deterministic 512-example train/test subsets over six
epochs using the publication model's ten simulation steps;
they are labelled as reduced validation rather than publication evidence. The
`smoke` profile likewise executes real model forwards on a tiny offline fixture.
Only `publication` replays the committed long-run
[`data/fig6_devicefixed_data.json`](data/fig6_devicefixed_data.json) archive.
Its adjacent [schema note](data/fig6_devicefixed_data.README.md) documents the
three-seed, twelve-epoch record. Set `MNN_GENERATE_STATIC_ARCHIVE=1` only to
persist an explicitly selected live run. Alternate device and architecture
views in the first notebook are hidden by
default; set `MNN_SHOW_CONTEXT_FIGURES=1` to render them. The generated
single-neuron SNN schematic remains part of the default reading flow.

The homeostasis architecture-dependence panel is generated live by default. It
runs paired homeostasis-off/on `MSNN` experiments on genuine MNIST and
Fashion-MNIST samples and a paired `TemporalMCSNN` experiment on deterministic
moving-MNIST clips. Three reduced seeds produce the displayed deltas and
bootstrap intervals. The curated source fixture is under `data/fixtures`; the
published 20-seed aggregate remains a distinct `publication` replay and is
never expanded into synthetic seed observations.

The temporal notebook likewise runs live by default. It trains the real
`TemporalMSNN` and `TemporalMCSNN` implementations on deterministic reduced
samples from MNIST, N-MNIST, SHD, and DVS Gesture, evaluates the trained models
over retention delays, and runs sample-level `DeviceLeaky`/`AdaptiveLeaky`
store-recall trials. The event fixture contains genuine source samples framed
into six time bins; its deliberately small 12-per-class/8-per-class
repartition is documented in [`data/fixtures/README.md`](data/fixtures/README.md).
These outputs are reduced mechanism validation, not the manuscript's full
benchmark scores. To replay a provenance-complete full-sweep archive instead,
place it at `data/results/temporal_seed_archive.json`, set
`MNN_USE_TEMPORAL_ARCHIVE=1`, and use the `publication` profile. Missing or
aggregate-only seed evidence is rejected rather than synthesized.

Each notebook exposes the same controls near the top:

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

- `reduced` renders measured evidence and runs bounded sample-level validation
  experiments where practical.
- `publication` renders measured records and validated sample archives without
  launching expensive reruns; full archive generation is explicitly opt-in.
- `smoke` is the minimal offline profile used by automated checks.

Figures are always displayed inline. Nothing is written unless
`SAVE_FIGURES=True`; existing files and manuscript directories additionally
require explicit overwrite permission. Reference-only images and missing-data
placeholders cannot be exported as regenerated evidence.

## Representation data

N-MNIST and EEG representation figures run offline by default from committed,
class-balanced samples of the genuine upstream datasets. The notebook trains
the actual `TemporalMCSNN` and `MCSNN` models and extracts second-convolution
spikes live. The small repartitions and single seed are explicitly labelled
reduced validation.

Full source downloads are optional and never implicit. Install the `external`
extra and set both `MNN_ALLOW_DATA_DOWNLOADS=1` and
`MNN_DOWNLOAD_REPRESENTATION_DATA=1` to obtain them using the instructions and
download code inside the notebook. A provenance-complete long-run feature
archive can be replayed with `MNN_USE_REPRESENTATION_ARCHIVE=1`; malformed or
incomplete archives are rejected.

## Repository layout

```text
src/mnn_torch/   models, layers, device effects, and data interfaces
experiments/     publication-reproduction notebooks and their manifests
data/            measured fixtures and optional local datasets/caches
tests/           package and reproduction-contract checks
```

## Citation and licence

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The software is
released under the MIT licence; see [`LICENSE`](LICENSE).
