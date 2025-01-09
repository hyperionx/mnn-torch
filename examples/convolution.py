import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from snntorch import surrogate, backprop, functional as SF, utils
from mnn_torch.devices import load_SiOx_multistate
from mnn_torch.effects import compute_PooleFrenkel_parameters
from mnn_torch.models import MCSNN


def main():
    # Dataloader arguments
    batch_size = 128
    data_path = "./data"
    experimental_data = load_SiOx_multistate("./data/SiO_x-multistate-data.mat")
    (
        G_off,
        G_on,
        R,
        c,
        d_epsilon,
    ) = compute_PooleFrenkel_parameters(experimental_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define a transform for MNIST
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))
    ])

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    training_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

    spike_grad = surrogate.fast_sigmoid(slope=25)
    beta = 0.5
    num_steps = 25

    # Network Architecture parameters
    num_kernels = 5
    num_conv1 = 12
    num_conv2 = 64
    max_pooling = 2
    num_hidden = num_conv2 * 4 * 4  # Adjust according to your layer sizes
    num_outputs = 10

    # Memristive configuration (set "ideal" to False or True for different behavior)
    PF_config = {
        "k_V": 0.5,
        "ideal": False,
        "disturb_conductance": False,
        "G_off": G_off,
        "G_on": G_on,
        "R": R,
        "c": c,
        "d_epsilon": d_epsilon,
    }

    # Initialize model
    net = MCSNN(
        beta=beta,
        spike_grad=spike_grad,
        batch_size=batch_size,
        num_kernels=num_kernels,
        num_conv1=num_conv1,
        num_conv2=num_conv2,
        max_pooling=max_pooling,
        num_hidden=num_hidden,
        num_outputs=num_outputs,
        memristive_config=PF_config
    ).to(device)

    loss_fn = SF.ce_rate_loss()

    # Forward pass function
    def forward_pass(net, num_steps, data):
        mem_rec = []
        spk_rec = []
        utils.reset(net)  # resets hidden states for all LIF neurons in net

        for step in range(num_steps):
            spk_out, mem_out = net(data)
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)

    # Batch accuracy function
    def batch_accuracy(loader, net, num_steps):
        with torch.no_grad():
            total = 0
            acc = 0
            net.eval()

            for data, targets in loader:
                data = data.to(device)
                targets = targets.to(device)
                spk_rec, _ = forward_pass(net, num_steps, data)

                acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
                total += spk_rec.size(1)

        return acc / total

    # Optimizer setup
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
    num_epochs = 1
    loss_hist = []
    test_acc_hist = []
    counter = 0

    # Outer training loop
    for epoch in range(num_epochs):
        # Training loop
        net.train()
        for data, targets in training_loader:
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            spk_rec, _ = forward_pass(net, num_steps, data)

            # Compute loss
            loss_val = loss_fn(spk_rec, targets)

            # Backprop and optimization
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Record loss history
            loss_hist.append(loss_val.item())

            # Evaluation on the test set every 50 steps
            if counter % 50 == 0:
                test_acc = batch_accuracy(validation_loader, net, num_steps)
                print(f"Iteration {counter}, Test Accuracy: {test_acc * 100:.2f}%")
                test_acc_hist.append(test_acc.item())

            counter += 1

    pass  # Training is complete


if __name__ == "__main__":
    main()
