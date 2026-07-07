# Experiments (Reproducibility Snapshot)

This directory contains the definitive reproducibility notebooks for the `mnn-torch` package, as described in the thesis appendix. These notebooks run the core experiments and regenerate the thesis figures.

## The Notebooks

### 1. `examples.ipynb`
Trains the memristive MSNN/MCSNN on MNIST with the Poole-Frenkel device forward map.

### 2. `homeostasis.ipynb`
Runs the fault-recovery sweep (with the homeostatic regulariser on versus off under the stuck-at prior) and reports the recovery gap by fault polarity.

### 3. `temporal_storerecall.ipynb`
Replays the temporal-benchmark capability and retention grids and the store-recall network, showing that the device membrane holds a delayed match as well as the adaptive neuron.

### 4. `REPRODUCE_ALL.ipynb`
A master runner that executes all of the topic notebooks automatically.

## Running the Experiments

By default, the notebooks will try to replay committed grids from `data/results/`. To run quick previews (in-kernel, reduced seeds), ensure the `QUICK = True` flag is set at the top of the notebooks.

If you set `QUICK = False`, the notebooks strictly require the generated data files. You can generate these missing grids yourself from the command line (make sure to run these from the `mnn_torch` conda environment, which has the necessary CUDA configuration):

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

*Note: The commands above run the default "quick" grids which take only a few minutes. To run the heavy 20-seed sweeps that produce the publication-quality figures, append the `--full` flag to any of these commands.*
