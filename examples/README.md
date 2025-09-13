# MNN-Torch Examples

This directory contains examples and tutorials for using MNN-Torch.

## Interactive Tutorials

### 🚀 Quick Start: `tutorial_quick_test.py`
**Recommended for first-time users**

A simplified tutorial that tests all basic functionality without full training loops:
- CUDA availability testing
- Model creation and forward passes
- Single training step verification
- MNIST data loading

**How to run:**
```bash
# In Cursor IDE:
# 1. Open examples/tutorial_quick_test.py
# 2. Use Ctrl+Shift+P → "Python: Run Selection/Line in Python Terminal"
# 3. Run each cell (marked with # %%) individually

# Or run the entire file:
uv run python examples/tutorial_quick_test.py
```

### 📚 Full Tutorial: `tutorial_interactive.py`
**Complete tutorial with full training**

The complete tutorial converted from the Jupyter notebook:
- Full training loops for both MSNN and MCSNN
- Performance comparisons with/without homeostasis dropout
- Comprehensive plotting and analysis
- Results summary

**How to run:**
```bash
# In Cursor IDE:
# 1. Open examples/tutorial_interactive.py
# 2. Run each cell individually using the cell markers (# %%)

# Note: This will take several minutes to complete due to training
```

### 📓 Original Jupyter: `tutorial.ipynb`
**Interactive Jupyter notebook**

The original Jupyter notebook for interactive exploration:
```bash
# Start Jupyter notebook
uv run jupyter notebook

# Then open examples/tutorial.ipynb in your browser
```

## Other Examples

### `convolution.py`
Standalone MNIST classification script with memristive CNN.

### `biosignal.ipynb`
Biosignal processing example with memristive neural networks.

### `ITO.ipynb`
ITO memristor characterization and modeling.

## Running Examples in Cursor IDE

### Method 1: Cell-by-Cell Execution
1. Open any `.py` file with cell markers (`# %%`)
2. Place cursor in a cell
3. Use `Ctrl+Shift+P` → "Python: Run Selection/Line in Python Terminal"
4. Or use the "Run Cell" button if available

### Method 2: Full File Execution
```bash
# Run entire file
uv run python examples/tutorial_quick_test.py
```

### Method 3: Jupyter Notebook
```bash
# Start Jupyter
uv run jupyter notebook

# Open examples/tutorial.ipynb
```

## Prerequisites

Make sure you have the UV environment set up:

```bash
# Install dependencies
uv sync

# Activate environment
uv shell
```

## Troubleshooting

### CUDA Issues
```bash
# Test CUDA availability
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Import Errors
```bash
# Make sure you're in the virtual environment
uv shell

# Reinstall if needed
uv sync --reinstall
```

### Path Issues
The examples automatically detect the correct data path:
- If running from project root: `../data/`
- If running from examples folder: `data/`

## Expected Results

### Quick Test Tutorial
- ✅ CUDA detection and device info
- ✅ Model creation (MSNN: 79,510 params, MCSNN: 29,826 params)
- ✅ Forward passes on GPU
- ✅ Training step completion
- ✅ MNIST data loading

### Full Tutorial
- ✅ Training both MSNN and MCSNN models
- ✅ Performance comparisons
- ✅ Accuracy plots and analysis
- ✅ Final accuracy: ~70-80% on MNIST

## Next Steps

After running the tutorials:

1. **Experiment with parameters**: Modify hyperparameters in the config
2. **Try different datasets**: Adapt the code for other datasets
3. **Explore architectures**: Create custom memristive layers
4. **Read the documentation**: Check `docs/` for detailed API reference

## Need Help?

- Check the main [README.md](../README.md) for installation instructions
- Open an [issue](https://github.com/hyperionx/mnn-torch/issues) for bugs
- Join [discussions](https://github.com/hyperionx/mnn-torch/discussions) for questions
