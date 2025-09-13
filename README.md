# MNN-Torch

MNN-Torch is a PyTorch-based implementation for Memristor-based Neural Networks (MNN). The project enables the construction and training of spiking neural networks (SNNs) utilizing memristive components for more efficient neural computation.

## Features

- **Memristive Neural Networks**: Implementation of memristive layers for enhanced performance
- **Spiking Neural Networks**: Full SNN support with SNNTorch integration
- **PyTorch Integration**: Seamless integration with standard PyTorch layers and workflows
- **CUDA Support**: Optimized for modern GPUs including RTX 5090 with CUDA 12.8
- **Flexible Architecture**: Modular design for custom neural network architectures
- **Experimental Data Support**: Built-in support for memristive device characterization data

## Quick Start

### Prerequisites

- Python >= 3.10
- CUDA-capable GPU (recommended for best performance)
- [UV package manager](https://docs.astral.sh/uv/) (recommended)

### 1. Install UV (if not already installed)

```bash
# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/hyperionx/mnn-torch.git
cd mnn-torch

# Install all dependencies (including CUDA-enabled PyTorch)
uv sync

# Activate the virtual environment
uv shell
```

### 3. Verify CUDA Installation

```bash
# Test CUDA availability
uv run python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### 4. Run the Tutorial

```bash
# Start Jupyter notebook
uv run jupyter notebook

# Then open examples/tutorial.ipynb in your browser
```

## Alternative Installation Methods

### Using pip (without UV)

```bash
# Install PyTorch with CUDA 12.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install the package in development mode
pip install -e .

# Install additional dependencies
pip install jupyter matplotlib numpy scikit-learn snntorch scipy
```

## Basic Usage

### Simple Example

```python
import torch
from mnn_torch.models import MSNN
from mnn_torch.devices import load_SiOx_multistate
from mnn_torch.effects import compute_PooleFrenkel_parameters

# Load experimental memristive data
experimental_data = load_SiOx_multistate("./data/SiO_x-multistate-data.mat")
G_off, G_on, R, c, d_epsilon = compute_PooleFrenkel_parameters(experimental_data)

# Configure memristive parameters
memristive_config = {
    "ideal": False,
    "G_off": G_off,
    "G_on": G_on,
    "R": R,
    "c": c,
    "d_epsilon": d_epsilon,
    "disturb_conductance": True,
    "homeostasis_dropout": True,
    "homeostasis_threshold": 10,
}

# Create memristive spiking neural network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Fully connected memristive SNN
net = MSNN(
    num_inputs=784,      # MNIST input size
    num_hidden=100,      # Hidden layer size
    num_outputs=10,      # Number of classes
    num_steps=10,        # Time steps
    beta=0.95,           # LIF neuron decay
    memristive_config=memristive_config,
).to(device)

# Test forward pass
x = torch.rand(64, 784).to(device)
spk_rec, mem_rec = net(x)
print(f"Output shape: {spk_rec.shape}")
```

### Running Examples

The easiest way to get started is with the interactive tutorial:

```bash
# Start Jupyter notebook
uv run jupyter notebook

# Open examples/tutorial.ipynb
```

Available examples in the `examples/` directory:

- **`tutorial.ipynb`** - Interactive tutorial with MNIST classification
  - Fully connected memristive SNN
  - Convolutional memristive SNN
  - Performance comparison with/without homeostasis dropout
- **`convolution.py`** - Standalone MNIST classification script
- **`biosignal.ipynb`** - Biosignal processing example
- **`ITO.ipynb`** - ITO memristor characterization

### Quick Test

To verify everything is working:

```bash
# Test CUDA and basic functionality
uv run python -c "
import torch
from mnn_torch.models import MSNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
config = {'ideal': False, 'k_V': 0.5, 'G_off': 1e-6, 'G_on': 1e-3, 'R': 1.0, 'c': 1.0, 'd_epsilon': 1.0, 'disturb_conductance': True, 'disturb_mode': 'fixed', 'disturbance_probability': 0.8, 'homeostasis_dropout': True, 'homeostasis_threshold': 10}
net = MSNN(784, 100, 10, 10, 0.95, config).to(device)
x = torch.rand(64, 784).to(device)
spk_rec, mem_rec = net(x)
print('✓ MNN-Torch is working correctly!')
"
```

## Development

### Setting up Development Environment

```bash
# Install with development dependencies
uv sync --extra dev

# Activate the environment
uv shell
```

### Development Commands

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/mnn_torch --cov-report=html

# Format code
uv run black src/ tests/

# Sort imports
uv run isort src/ tests/

# Type checking
uv run mypy src/

# Linting
uv run flake8 src/ tests/
```

### Project Structure

```
mnn-torch/
├── src/mnn_torch/          # Main package
│   ├── devices.py          # Device data loading
│   ├── effects.py          # Memristive effects
│   ├── layers.py           # Memristive layers
│   ├── models.py           # Neural network models
│   └── utils.py            # Utility functions
├── examples/               # Example notebooks and scripts
│   ├── tutorial.ipynb      # Interactive tutorial
│   ├── convolution.py      # MNIST classification
│   ├── biosignal.ipynb     # Biosignal processing
│   └── ITO.ipynb          # ITO memristor characterization
├── tests/                  # Test suite
├── data/                   # Experimental data
└── docs/                   # Documentation
```

### Adding New Features

1. Create a feature branch: `git checkout -b feature/new-feature`
2. Make your changes in `src/mnn_torch/`
3. Add tests in `tests/`
4. Run tests: `uv run pytest`
5. Format code: `uv run black src/ tests/`
6. Submit a pull request

## Troubleshooting

### Common Issues

**CUDA not detected:**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
uv run python -c "import torch; print(torch.cuda.is_available())"
```

**Import errors:**
```bash
# Make sure you're in the virtual environment
uv shell

# Reinstall dependencies
uv sync --reinstall
```

**Jupyter notebook issues:**
```bash
# Install Jupyter in the UV environment
uv add jupyter

# Start Jupyter from the project directory
uv run jupyter notebook
```

**Memory issues with large models:**
- Reduce batch size in the configuration
- Use `torch.cuda.empty_cache()` to clear GPU memory
- Consider using gradient accumulation for large models

### Getting Help

- Check the [examples/](examples/) directory for working code
- Open an [issue](https://github.com/hyperionx/mnn-torch/issues) for bugs
- Join discussions in [GitHub Discussions](https://github.com/hyperionx/mnn-torch/discussions)

## Documentation

Full documentation is available at [mnn-torch.readthedocs.io](https://mnn-torch.readthedocs.io).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.rst](CONTRIBUTING.rst) for guidelines.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `uv run pytest`
5. Format your code: `uv run black src/ tests/`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## License

MIT License - see [LICENSE.txt](LICENSE.txt) for details.

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

## Acknowledgments

- Built on top of [PyTorch](https://pytorch.org/) and [SNNTorch](https://snntorch.readthedocs.io/)
- Experimental data from memristive device characterizations
- Inspired by research in neuromorphic computing and memristive neural networks