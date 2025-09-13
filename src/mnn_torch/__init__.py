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
from .effects import compute_PooleFrenkel_parameters
from .layers import MemristiveLinearLayer, MemristiveConv2d, HomeostasisDropout

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
    
    # Layers
    "MemristiveLinearLayer",
    "MemristiveConv2d",
    "HomeostasisDropout",
]
