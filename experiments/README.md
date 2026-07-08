# Experiments (Reproducibility Snapshot)

This directory contains the definitive reproducibility notebooks for the `mnn-torch` package. These notebooks run the core experiments and regenerate the publication-facing figures.

## The Notebooks

### 1. `examples.ipynb`
Trains the memristive MSNN/MCSNN on MNIST with the Poole-Frenkel device forward map.

### 2. `homeostasis.ipynb`
Runs the fault-recovery sweep (with the homeostatic regulariser on versus off under the stuck-at prior) and reports the recovery gap by fault polarity.

### 3. `temporal_storerecall.ipynb`
Runs reduced-budget temporal-memory, retention, class-structure, and store-recall computations by default, with an explicit full-sweep cache-rendering mode for heavier outputs.

### 4. `REPRODUCE_ALL.ipynb`
A master runner that executes all of the topic notebooks automatically.

## Running the Experiments

By default, the notebooks run in `RESULT_MODE = "live"` and compute reduced-budget versions of the result panels directly from the package code.

Set `RESULT_MODE = "full_sweep_cache"` only when you want to render committed outputs from heavier publication-scale sweeps. Those cells print the cache source/provenance and include commented command patterns for regenerating the data.

You can generate the available diagnostic grids from the command line (make sure to run these from the `mnn_torch` conda environment, which has the necessary CUDA configuration):

```bash
# 1. Generate the homeostasis gate sweep
conda run -n mnn_torch python -m mnn_torch.training --gate

# 2. Generate the dose-response evaluation for different fault rates
conda run -n mnn_torch python -m mnn_torch.training --dose --rate 0.0
conda run -n mnn_torch python -m mnn_torch.training --dose --rate 0.2
conda run -n mnn_torch python -m mnn_torch.training --dose --rate 0.4

# 3. Generate the forward map vs. sampler ablations (Isolate Collapse)
conda run -n mnn_torch python -m mnn_torch.training --isolate --map pf --mode device
conda run -n mnn_torch python -m mnn_torch.training --isolate --map pf --mode fixed
conda run -n mnn_torch python -m mnn_torch.training --isolate --map ohmic --mode device
conda run -n mnn_torch python -m mnn_torch.training --isolate --map ohmic --mode fixed
```

*Note: The commands above run the default quick diagnostic grids. To run heavier publication-scale diagnostic grids, append the `--full` flag where supported. Notebook figures should be generated from code in live mode; cached arrays are an explicit full-sweep rendering path, not the default.*
