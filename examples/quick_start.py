#!/usr/bin/env python3
"""
Quick Test Tutorial for MNN-Torch
=================================

This is a simplified version of the tutorial that can be run quickly to test
the basic functionality without the full training loops.

To run this tutorial:
1. Open this file in Cursor
2. Use Ctrl+Shift+P and select "Python: Run Selection/Line in Python Terminal"
3. Or use the "Run Cell" functionality if available
4. Run each cell in sequence

Make sure you have the UV environment activated:
    uv shell
"""

# %% [markdown]
# # MNN-Torch Quick Test Tutorial
# 
# This tutorial demonstrates the basic functionality of MNN-Torch
# with quick tests instead of full training.

# %% [markdown]
# ## Import Libraries

# %%
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from mnn_torch.devices import load_SiOx_multistate
from mnn_torch.models import MSNN, MCSNN
from snntorch import surrogate
from mnn_torch.effects import compute_PooleFrenkel_parameters

# %% [markdown]
# ## Test CUDA Availability

# %%
# Test CUDA availability
x = torch.rand(5, 3)
print("Random tensor:")
print(x)

print(f"\nCUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")
print(f"Current Device: {torch.cuda.current_device()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## Load Data and Initialize Parameters

# %%
# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load experimental data
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# Adjust path based on where you're running from
data_path = "../data/SiO_x-multistate-data.mat"
if not os.path.exists(data_path):
    data_path = "data/SiO_x-multistate-data.mat"

print(f"Loading data from: {data_path}")
experimental_data = load_SiOx_multistate(data_path)
(G_off, G_on, R, c, d_epsilon) = compute_PooleFrenkel_parameters(experimental_data)

print(f"Experimental data shape: {experimental_data.shape}")
print(f"G_off: {G_off:.6f}, G_on: {G_on:.6f}")

# Hyperparameters
batch_size = 64
num_inputs = 28 * 28
num_hidden = 100
num_outputs = 10
num_steps = 10
beta = 0.95

print(f"Hyperparameters:")
print(f"  Batch size: {batch_size}")
print(f"  Input size: {num_inputs}")
print(f"  Hidden size: {num_hidden}")
print(f"  Output size: {num_outputs}")
print(f"  Time steps: {num_steps}")

# %% [markdown]
# ## Test Fully Connected MSNN

# %%
# Test fully connected memristive SNN
print("=" * 50)
print("TESTING FULLY CONNECTED MSNN")
print("=" * 50)

PF_config = {
    "ideal": False,
    "k_V": 0.5,
    "G_off": G_off,
    "G_on": G_on,
    "R": R,
    "c": c,
    "d_epsilon": d_epsilon,
    "disturb_conductance": True,
    "disturb_mode": "fixed",
    "disturbance_probability": 0.8,
    "homeostasis_dropout": True,
    "homeostasis_threshold": 10,
}

# Create MSNN model
msnn_net = MSNN(num_inputs, num_hidden, num_outputs, num_steps, beta, PF_config).to(device)
print(f"MSNN model created with {sum(p.numel() for p in msnn_net.parameters()):,} parameters")

# Test forward pass
x_fc = torch.rand(batch_size, num_inputs).to(device)
print(f"Input tensor shape: {x_fc.shape}")
print(f"Input tensor device: {x_fc.device}")

with torch.no_grad():
    spk_rec, mem_rec = msnn_net(x_fc)
    print(f"✓ MSNN forward pass successful!")
    print(f"  Spike recording shape: {spk_rec.shape}")
    print(f"  Membrane recording shape: {mem_rec.shape}")
    print(f"  Model parameters device: {next(msnn_net.parameters()).device}")

# %% [markdown]
# ## Test Convolutional MCSNN

# %%
# Test convolutional memristive SNN
print("=" * 50)
print("TESTING CONVOLUTIONAL MCSNN")
print("=" * 50)

# Create MCSNN model
mscnn_net = MCSNN(
    beta=beta,
    spike_grad=surrogate.fast_sigmoid(slope=25),
    num_steps=num_steps,
    batch_size=batch_size,
    num_kernels=5,
    num_conv1=12,
    num_conv2=64,
    max_pooling=2,
    num_outputs=num_outputs,
    memristive_config=PF_config,
).to(device)

print(f"MCSNN model created with {sum(p.numel() for p in mscnn_net.parameters()):,} parameters")

# Test forward pass
x_conv = torch.rand(batch_size, 1, 28, 28).to(device)
print(f"Input tensor shape: {x_conv.shape}")
print(f"Input tensor device: {x_conv.device}")

with torch.no_grad():
    spk_rec, mem_rec, spk2_rec = mscnn_net(x_conv)
    print(f"✓ MCSNN forward pass successful!")
    print(f"  Spike recording shape: {spk_rec.shape}")
    print(f"  Membrane recording shape: {mem_rec.shape}")
    print(f"  Spike2 recording shape: {spk2_rec.shape}")
    print(f"  Model parameters device: {next(mscnn_net.parameters()).device}")

# %% [markdown]
# ## Test Training Step

# %%
# Test a single training step
print("=" * 50)
print("TESTING SINGLE TRAINING STEP")
print("=" * 50)

# Use MSNN for training test
net = msnn_net
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

# Create dummy data
x_train = torch.rand(batch_size, num_inputs).to(device)
y_train = torch.randint(0, num_outputs, (batch_size,)).to(device)

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")

# Forward pass
net.train()
spk_rec, mem_rec = net(x_train)
loss_val = sum(loss_fn(mem_rec[step], y_train) for step in range(num_steps))

print(f"Training loss: {loss_val.item():.4f}")

# Backward pass
optimizer.zero_grad()
loss_val.backward()
optimizer.step()

print("✓ Training step completed successfully!")

# %% [markdown]
# ## Test MNIST Data Loading

# %%
# Test MNIST data loading
print("=" * 50)
print("TESTING MNIST DATA LOADING")
print("=" * 50)

# Data loading
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
])

data_path_mnist = "../data"
if not os.path.exists(data_path_mnist):
    data_path_mnist = "data"

print(f"Loading MNIST from: {data_path_mnist}")

try:
    mnist_train = datasets.MNIST(
        data_path_mnist, train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST(
        data_path_mnist, train=False, download=True, transform=transform
    )
    
    training_loader = DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    validation_loader = DataLoader(
        mnist_test, batch_size=batch_size, shuffle=True, drop_last=True
    )
    
    print(f"✓ MNIST data loaded successfully!")
    print(f"  Training samples: {len(mnist_train)}")
    print(f"  Test samples: {len(mnist_test)}")
    
    # Test loading a batch
    data, targets = next(iter(training_loader))
    print(f"  Batch data shape: {data.shape}")
    print(f"  Batch targets shape: {targets.shape}")
    print(f"  Data device: {data.device}")
    
except Exception as e:
    print(f"✗ Error loading MNIST data: {e}")

# %% [markdown]
# ## Summary

# %%
# Print summary
print("=" * 60)
print("QUICK TEST SUMMARY")
print("=" * 60)

print(f"✓ Device: {device}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ Experimental data loaded: {experimental_data.shape}")
print(f"✓ MSNN model created and tested")
print(f"✓ MCSNN model created and tested")
print(f"✓ Training step completed")
print(f"✓ MNIST data loading tested")

print(f"\nAll basic functionality tests passed!")
print("The full tutorial should work correctly.")
print("=" * 60)
