"""
MNN-Torch: Memristive Neural Networks with PyTorch

A library for implementing memristive neural networks using PyTorch,
featuring spiking neural networks with memristive components for
enhanced performance and hardware acceleration.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "mnn-torch"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

# Import main classes and functions
from .models import MSNN, MCSNN, BaseSNN, LayerBuilder
from .devices import load_SiOx_multistate, load_SiOx_curves, clean_experimental_data
from .effects import (
    compute_PooleFrenkel_parameters,
    compute_PooleFrenkel_regression_parameters,
    compute_PooleFrenkel_current_torch,
    sample_PooleFrenkel_parameters_torch,
)
from .layers import (
    MemristiveLinearLayer,
    MemristiveConv2d,
    HomeostasisDropout,
    HomeostaticRegulariser,
)
from . import paths
from .paths import data_dir, results_dir, device_data_mat, save_result, load_result
from . import data, training
from .data import mnist_loaders, ensure_mnist, prep_bci2a
from .training import (
    run_condition,
    gate_homeostasis_sweep,
    dose_response,
    isolate_collapse,
    precompute_device_params,
)

# Define what gets imported with "from mnn_torch import *"
__all__ = [
    # Version
    "__version__",
    
    # Models
    "MSNN",
    "MCSNN", 
    "BaseSNN",
    "LayerBuilder",
    
    # Device data loading
    "load_SiOx_multistate",
    "load_SiOx_curves", 
    "clean_experimental_data",
    
    # Memristive effects
    "compute_PooleFrenkel_parameters",
    "compute_PooleFrenkel_regression_parameters",
    "compute_PooleFrenkel_current_torch",
    "sample_PooleFrenkel_parameters_torch",

    # Layers
    "MemristiveLinearLayer",
    "MemristiveConv2d",
    "HomeostasisDropout",
    "HomeostaticRegulariser",

    # Paths / data-grid resolution (single source of truth for data/)
    "paths",
    "data_dir",
    "results_dir",
    "device_data_mat",
    "save_result",
    "load_result",

    # Data loading / preparation
    "data",
    "mnist_loaders",
    "ensure_mnist",
    "prep_bci2a",

    # Training / homeostasis fault-recovery studies
    "training",
    "run_condition",
    "gate_homeostasis_sweep",
    "dose_response",
    "isolate_collapse",
    "precompute_device_params",
]
