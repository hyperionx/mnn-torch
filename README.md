# MNN-Torch

MNN-Torch is a PyTorch-based implementation for Memristor-based Neural Networks (MNN). The project enables the construction and training of spiking neural networks (SNNs) utilizing memristive components for more efficient neural computation.

## Features

- **Memristive Neural Networks**: Implementation of memristive layers for enhanced performance
- **Spiking Neural Networks**: Full SNN support with SNNTorch integration
- **PyTorch Integration**: Seamless integration with standard PyTorch layers and workflows
- **CUDA Support**: Optimized for modern GPUs
- **Flexible Architecture**: Modular design for custom neural network architectures
- **Experimental Data Support**: Built-in support for memristive device characterization data

## Quick Start (Conda)

It is highly recommended to use the provided Conda environment which guarantees the correct CUDA toolkits and dependencies for the PyTorch models.

### 1. Clone the repository

```bash
git clone https://github.com/hyperionx/mnn-torch.git
cd mnn-torch
```

### 2. Activate Environment and Install

If you have a Conda environment (e.g. `mnn_torch` with Python 3.10+ and PyTorch+CUDA installed), activate it and install the library in development mode:

```bash
conda activate mnn_torch
pip install -e .
pip install jupyter matplotlib numpy scikit-learn snntorch scipy
```

### 3. Verify Installation

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

## Reproducing Publication Experiments

The `experiments/` directory contains the definitive reproducibility notebooks for the `mnn-torch` package.

By default, these notebooks run in `RESULT_MODE = "live"` and compute reduced-budget versions of the result panels directly from the package code.

To render committed outputs from heavier publication-scale sweeps, set `RESULT_MODE = "full_sweep_cache"`. Cache-rendering cells print the source file/provenance and keep the regeneration command pattern commented nearby for examiners who want to rerun the full sweep.

If a grid is missing, run the command-line scripts below to rebuild the quick versions locally.

To generate the missing grids locally, run the following commands from your active `mnn_torch` environment:

```bash
# 1. Generate the array-scale homeostasis fault-recovery grids
python -m mnn_torch.training --gate

# 2. Generate the dose-response slices
python -m mnn_torch.training --dose --rate 0.0
python -m mnn_torch.training --dose --rate 0.2
python -m mnn_torch.training --dose --rate 0.4

# 3. Generate the forward map vs. sampler ablations
python -m mnn_torch.training --isolate --map pf --mode device
python -m mnn_torch.training --isolate --map pf --mode fixed
python -m mnn_torch.training --isolate --map ohmic --mode device
python -m mnn_torch.training --isolate --map ohmic --mode fixed
```

*Note: The commands above execute the default quick diagnostic grids. To run heavier publication-scale diagnostic grids, append the `--full` flag where supported. Notebook figures should be generated from code in live mode; cached arrays are an explicit full-sweep rendering path, not the default.*

### Running the Notebooks

Once the data is generated, start your Jupyter server:

```bash
jupyter notebook
```

Navigate to `experiments/` and open:
- `REPRODUCE_ALL.ipynb`: Master runner for all topic notebooks
- `examples.ipynb`: Basic model training with device prior
- `homeostasis.ipynb`: Fault-recovery sweep
- `temporal_storerecall.ipynb`: Temporal benchmark capability and retention grids

## Project Structure

```
mnn-torch/
├── src/mnn_torch/          # Main package
│   ├── data.py             # Data loading utilities
│   ├── devices.py          # Device data loading
│   ├── effects.py          # Memristive effects
│   ├── layers.py           # Memristive layers
│   ├── models.py           # Neural network models
│   ├── paths.py            # Path resolution
│   └── training.py         # Homeostasis and fault-recovery sweep logic
├── experiments/            # Reproducibility notebooks
│   ├── README.md           # Instructions for experiments
│   ├── REPRODUCE_ALL.ipynb # Master runner notebook
│   ├── examples.ipynb      # MSNN/MCSNN training example
│   ├── homeostasis.ipynb   # Fault recovery sweep
│   └── temporal_storerecall.ipynb # Temporal benchmark
├── data/                   # Experimental data and results
└── docs/                   # Documentation
```

## Citation

If you use MNN-Torch in your research, please cite:

```bibtex
@software{mnn_torch,
  title={MNN-Torch: Memristive Neural Networks with PyTorch},
  author={hyperionx},
  year={2024},
  url={https://github.com/hyperionx/mnn-torch}
}
```
