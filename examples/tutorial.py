#!/usr/bin/env python3
"""
Interactive Tutorial for MNN-Torch
==================================

This file contains the tutorial content from tutorial.ipynb formatted as Python cells
that can be run interactively in Cursor IDE. Each cell is marked with a comment
and can be executed individually.

To run this tutorial:
1. Open this file in Cursor
2. Use Ctrl+Shift+P and select "Python: Run Selection/Line in Python Terminal"
3. Or use the "Run Cell" functionality if available
4. Run each cell in sequence

Make sure you have the UV environment activated:
    uv shell
"""

# %% [markdown]
# # MNN-Torch Interactive Tutorial
# 
# This tutorial demonstrates how to use MNN-Torch for memristive neural networks
# with both fully connected and convolutional architectures.

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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
num_epochs = 1
num_inputs = 28 * 28
num_hidden = 100
num_outputs = 10
num_steps = 10
beta = 0.95
data_path_mnist = "../data"
if not os.path.exists(data_path_mnist):
    data_path_mnist = "data"
lr = 5e-4

print(f"Hyperparameters:")
print(f"  Batch size: {batch_size}")
print(f"  Epochs: {num_epochs}")
print(f"  Learning rate: {lr}")
print(f"  MNIST data path: {data_path_mnist}")

# Data loading
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
])

print("Loading MNIST dataset...")
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

print(f"Training samples: {len(mnist_train)}")
print(f"Test samples: {len(mnist_test)}")

# %% [markdown]
# ## Fully Connected Memristive SNN

# %%
# Memristive configuration function
def train_model_with_dropout(homeostasis_dropout=True):
    """Train a fully connected memristive SNN with or without homeostasis dropout."""
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
        "homeostasis_dropout": homeostasis_dropout,
        "homeostasis_threshold": 10,
    }

    # Initialize network
    net = MSNN(num_inputs, num_hidden, num_outputs, num_steps, beta, PF_config).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss_hist = []
    test_loss_hist = []
    test_acc_hist = []

    print(f"Training MSNN with homeostasis_dropout={homeostasis_dropout}")
    print(f"Model parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        for iter_counter, (data, targets) in enumerate(training_loader):
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            net.train()
            spk_rec, mem_rec = net(data.view(batch_size, -1))
            loss_val = sum(loss(mem_rec[step], targets) for step in range(num_steps))

            # Backward pass
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            loss_hist.append(loss_val.item())

            # Evaluate on validation set
            if iter_counter % 50 == 0:
                with torch.no_grad():
                    net.eval()
                    test_data, test_targets = next(iter(validation_loader))
                    test_data, test_targets = test_data.to(device), test_targets.to(device)

                    test_spk, test_mem = net(test_data.view(batch_size, -1))
                    test_loss = sum(
                        loss(test_mem[step], test_targets) for step in range(num_steps)
                    )
                    test_loss_hist.append(test_loss.item())

                    # Compute accuracy
                    _, idx = test_spk.sum(dim=0).max(1)
                    acc = (idx == test_targets).float().mean().item()
                    test_acc_hist.append(acc)

                    print(
                        f"Epoch {epoch}, Iteration {iter_counter}\n"
                        f"Train Loss: {loss_val.item():.2f}, Test Loss: {test_loss.item():.2f}, "
                        f"Test Accuracy: {acc * 100:.2f}%"
                    )

    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    return loss_hist, test_loss_hist, test_acc_hist

# %% [markdown]
# ## Train Fully Connected Models

# %%
# Train and collect data for both configurations
print("=" * 60)
print("TRAINING FULLY CONNECTED MEMRISTIVE SNN")
print("=" * 60)

msnn_loss_hist_dropout = []
msnn_test_loss_hist_dropout = []
msnn_test_acc_hist_dropout = []

msnn_loss_hist_no_dropout = []
msnn_test_loss_hist_no_dropout = []
msnn_test_acc_hist_no_dropout = []

# Train with homeostasis dropout
print("\n1. Training with homeostasis dropout...")
msnn_loss_hist_dropout, msnn_test_loss_hist_dropout, msnn_test_acc_hist_dropout = train_model_with_dropout(homeostasis_dropout=True)

# Train without homeostasis dropout
print("\n2. Training without homeostasis dropout...")
msnn_loss_hist_no_dropout, msnn_test_loss_hist_no_dropout, msnn_test_acc_hist_no_dropout = train_model_with_dropout(homeostasis_dropout=False)

# %% [markdown]
# ## Plot Fully Connected Results

# %%
# Plotting the results for fully connected networks
plt.figure(figsize=(12, 6))

# Loss comparison
plt.subplot(1, 2, 1)
plt.plot(msnn_test_loss_hist_dropout, label='With Homeostasis Dropout', color='blue')
plt.plot(msnn_test_loss_hist_no_dropout, label='Without Homeostasis Dropout', color='red')
plt.title('Fully Connected: Test Loss Comparison')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy comparison
plt.subplot(1, 2, 2)
plt.plot(msnn_test_acc_hist_dropout, label='With Homeostasis Dropout', color='blue')
plt.plot(msnn_test_acc_hist_no_dropout, label='Without Homeostasis Dropout', color='red')
plt.title('Fully Connected: Test Accuracy Comparison')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Convolutional Memristive SNN

# %%
# Convolutional memristive configuration function
def train_conv_model_with_dropout(homeostasis_dropout=True):
    """Train a convolutional memristive SNN with or without homeostasis dropout."""
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
        "homeostasis_dropout": homeostasis_dropout,
        "homeostasis_threshold": 10,
    }

    # Initialize the network
    net = MCSNN(
        beta=beta,
        spike_grad=surrogate.fast_sigmoid(slope=25),
        num_steps=num_steps,
        batch_size=batch_size,
        num_kernels=5,
        num_conv1=12,
        num_conv2=64,
        max_pooling=2,
        num_outputs=10,
        memristive_config=PF_config,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss_hist = []
    test_loss_hist = []
    test_acc_hist = []
    
    all_preds = []
    all_targets = []

    print(f"Training MCSNN with homeostasis_dropout={homeostasis_dropout}")
    print(f"Model parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        for iter_counter, (data, targets) in enumerate(training_loader):
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            net.train()
            spk_rec, mem_rec, spk2_rec = net(data)
            loss_val = sum(loss_fn(mem_rec[step], targets) for step in range(num_steps))

            # Backward pass
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            loss_hist.append(loss_val.item())

            # Evaluate on validation set
            if iter_counter % 50 == 0:
                with torch.no_grad():
                    net.eval()
                    test_data, test_targets = next(iter(validation_loader))
                    test_data, test_targets = test_data.to(device), test_targets.to(device)

                    test_spk, test_mem, test_spk2 = net(test_data)
                    test_loss = sum(loss_fn(test_mem[step], test_targets) for step in range(num_steps))
                    test_loss_hist.append(test_loss.item())

                    # Compute accuracy and accumulate predictions
                    _, idx = test_spk.sum(dim=0).max(1)
                    acc = (idx == test_targets).float().mean().item()
                    test_acc_hist.append(acc)

                    all_preds.extend(idx.cpu().numpy())
                    all_targets.extend(test_targets.cpu().numpy())

                    print(
                        f"Epoch {epoch}, Iteration {iter_counter}\n"
                        f"Train Loss: {loss_val.item():.2f}, Test Loss: {test_loss.item():.2f}, "
                        f"Test Accuracy: {acc * 100:.2f}%"
                    )

    # Compute and display the confusion matrix
    if len(all_preds) > 0:
        cm = confusion_matrix(all_targets, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix (Homeostasis Dropout: {homeostasis_dropout})')
        plt.show()

    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    return loss_hist, test_loss_hist, test_acc_hist

# %% [markdown]
# ## Train Convolutional Models

# %%
# Train and collect data for both configurations
print("=" * 60)
print("TRAINING CONVOLUTIONAL MEMRISTIVE SNN")
print("=" * 60)

mscnn_loss_hist_dropout = []
mscnn_test_loss_hist_dropout = []
mscnn_test_acc_hist_dropout = []

mscnn_loss_hist_no_dropout = []
mscnn_test_loss_hist_no_dropout = []
mscnn_test_acc_hist_no_dropout = []

# Train with homeostasis dropout
print("\n1. Training with homeostasis dropout...")
mscnn_loss_hist_dropout, mscnn_test_loss_hist_dropout, mscnn_test_acc_hist_dropout = train_conv_model_with_dropout(homeostasis_dropout=True)

# Train without homeostasis dropout
print("\n2. Training without homeostasis dropout...")
mscnn_loss_hist_no_dropout, mscnn_test_loss_hist_no_dropout, mscnn_test_acc_hist_no_dropout = train_conv_model_with_dropout(homeostasis_dropout=False)

# %% [markdown]
# ## Plot Convolutional Results

# %%
# Plotting the results for convolutional networks
plt.figure(figsize=(12, 6))

# Loss comparison
plt.subplot(1, 2, 1)
plt.plot(mscnn_test_loss_hist_dropout, label='With Homeostasis Dropout', color='blue')
plt.plot(mscnn_test_loss_hist_no_dropout, label='Without Homeostasis Dropout', color='red')
plt.title('Convolutional: Test Loss Comparison')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy comparison
plt.subplot(1, 2, 2)
plt.plot(mscnn_test_acc_hist_dropout, label='With Homeostasis Dropout', color='blue')
plt.plot(mscnn_test_acc_hist_no_dropout, label='Without Homeostasis Dropout', color='red')
plt.title('Convolutional: Test Accuracy Comparison')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Final Comparison: All Models

# %%
# Plotting the results - final comparison
plt.figure(figsize=(15, 6))

# Loss comparison
plt.subplot(1, 2, 1)
plt.plot([x * 0.1 for x in msnn_loss_hist_dropout], label='Fully-Connected + Dropout', alpha=0.7)
plt.plot([x * 0.1 for x in msnn_loss_hist_no_dropout], label='Fully-Connected - Dropout', alpha=0.7)
plt.plot([x * 0.5 for x in mscnn_loss_hist_dropout], label='Convolutional + Dropout', alpha=0.7)
plt.plot([x * 0.5 for x in mscnn_loss_hist_no_dropout], label='Convolutional - Dropout', alpha=0.7)
plt.title('Training Loss Comparison (All Models)')
plt.xlabel('Iterations')
plt.ylabel('Loss (scaled)')
plt.legend()
plt.grid(True)

# Accuracy comparison
plt.subplot(1, 2, 2)
plt.plot(msnn_test_acc_hist_dropout, label='Fully-Connected + Dropout', linestyle=':', linewidth=2)
plt.plot(msnn_test_acc_hist_no_dropout, label='Fully-Connected - Dropout', linestyle=':', linewidth=2)
plt.plot(mscnn_test_acc_hist_dropout, label='Convolutional + Dropout', color='#1f77b4', linestyle='--', linewidth=2)
plt.plot(mscnn_test_acc_hist_no_dropout, label='Convolutional - Dropout', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title('Test Accuracy Comparison (All Models)')
plt.xlabel('Evaluation Points')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary

# %%
# Print summary of results
print("=" * 80)
print("TUTORIAL SUMMARY")
print("=" * 80)

print(f"\nDevice used: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")

print(f"\nFully Connected MSNN Results:")
print(f"  With homeostasis dropout - Final accuracy: {msnn_test_acc_hist_dropout[-1]*100:.2f}%")
print(f"  Without homeostasis dropout - Final accuracy: {msnn_test_acc_hist_no_dropout[-1]*100:.2f}%")

print(f"\nConvolutional MCSNN Results:")
print(f"  With homeostasis dropout - Final accuracy: {mscnn_test_acc_hist_dropout[-1]*100:.2f}%")
print(f"  Without homeostasis dropout - Final accuracy: {mscnn_test_acc_hist_no_dropout[-1]*100:.2f}%")

print(f"\nBest performing model:")
best_fc = max(msnn_test_acc_hist_dropout[-1], msnn_test_acc_hist_no_dropout[-1])
best_conv = max(mscnn_test_acc_hist_dropout[-1], mscnn_test_acc_hist_no_dropout[-1])
if best_fc > best_conv:
    print(f"  Fully Connected MSNN: {best_fc*100:.2f}%")
else:
    print(f"  Convolutional MCSNN: {best_conv*100:.2f}%")

print(f"\nTutorial completed successfully!")
print("=" * 80)

# %%
# Optional: Save results for later analysis
results = {
    'msnn_dropout': {
        'loss': msnn_loss_hist_dropout,
        'test_loss': msnn_test_loss_hist_dropout,
        'test_acc': msnn_test_acc_hist_dropout
    },
    'msnn_no_dropout': {
        'loss': msnn_loss_hist_no_dropout,
        'test_loss': msnn_test_loss_hist_no_dropout,
        'test_acc': msnn_test_acc_hist_no_dropout
    },
    'mscnn_dropout': {
        'loss': mscnn_loss_hist_dropout,
        'test_loss': mscnn_test_loss_hist_dropout,
        'test_acc': mscnn_test_acc_hist_dropout
    },
    'mscnn_no_dropout': {
        'loss': mscnn_loss_hist_no_dropout,
        'test_loss': mscnn_test_loss_hist_no_dropout,
        'test_acc': mscnn_test_acc_hist_no_dropout
    }
}

print("Results saved in 'results' dictionary for further analysis.")
print("You can access individual results like: results['msnn_dropout']['test_acc']")
