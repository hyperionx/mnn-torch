import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from mnn_torch.devices import load_SiOx_multistate
from mnn_torch.models import MCSNN
from snntorch import surrogate, functional as SF, utils
from mnn_torch.effects import compute_PooleFrenkel_parameters


def main():
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load experimental data
    experimental_data = load_SiOx_multistate("./data/SiO_x-multistate-data.mat")
    (
        G_off,
        G_on,
        R,
        c,
        d_epsilon,
    ) = compute_PooleFrenkel_parameters(experimental_data)

    # Hyperparameters
    batch_size = 64
    num_epochs = 1
    num_steps = 200
    beta = 0.95
    data_path = "./data"
    lr = 5e-4

    # Memristive configuration
    PF_config = {
        "ideal": False,
        "k_V": 0.5,
        "G_off": G_off,
        "G_on": G_on,
        "R": R,
        "c": c,
        "d_epsilon": d_epsilon,
        "disturb_conductance": False,
        "disturb_mode": "fixed",
        "disturbance_probability": 0.1,
        "homeostasis_dropout": False,
        "homeostasis_threshold": 10,
    }

    # Data loading
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ]
    )
    mnist_train = datasets.MNIST(
        data_path, train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST(
        data_path, train=False, download=True, transform=transform
    )

    training_loader = DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    validation_loader = DataLoader(
        mnist_test, batch_size=batch_size, shuffle=True, drop_last=True
    )

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

    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        for iter_counter, (data, targets) in enumerate(training_loader):
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            net.train()
            spk_rec, mem_rec = net(data)
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
                    test_data, test_targets = test_data.to(device), test_targets.to(
                        device
                    )

                    test_spk, test_mem = net(test_data)
                    test_loss = sum(
                        loss_fn(test_mem[step], test_targets)
                        for step in range(num_steps)
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


if __name__ == "__main__":
    main()
