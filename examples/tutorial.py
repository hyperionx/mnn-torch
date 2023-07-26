import time
import numpy as np
import torch, torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

from mnn_torch.devices import load_SiOx_multistate
from mnn_torch.models import MSNN


def main():
    start_time = time.time()

    experimental_data = load_SiOx_multistate("./data/SiO_x-multistate-data.mat")

    # dataloader arguments
    batch_size = 128
    data_path = "./data"

    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Define a transform
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

    # train_indices = list(range(len(mnist_train)))
    # np.random.shuffle(train_indices)
    # train_sampler = SubsetRandomSampler(train_indices[:5000])

    # val_indices = list(range(len(mnist_test)))
    # np.random.shuffle(val_indices)
    # val_sampler = SubsetRandomSampler(val_indices[:1000])

    # # Create data loaders for our datasets; shuffle for training, not for validation
    # training_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, sampler=train_sampler)
    # validation_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, sampler=val_sampler)

    training_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    validation_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Network Architecture
    num_inputs = 28 * 28
    num_hidden = 1000
    num_outputs = 10

    # Temporal Dynamics
    num_steps = 25
    beta = 0.95

    net = MSNN(
        device,
        num_inputs,
        num_hidden,
        num_outputs,
        num_steps,
        beta,
        experimental_data=experimental_data,
        ideal=True,
    ).to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

    num_epochs = 1
    loss_hist = []
    test_loss_hist = []
    counter = 0

    def print_batch_accuracy(data, targets, train=False):
        output, _ = net(data.view(batch_size, -1))
        _, idx = output.sum(dim=0).max(1)
        acc = np.mean((targets == idx).detach().cpu().numpy())

        if train:
            print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
        else:
            print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

    def train_printer():
        print(f"--- {time.time() - start_time} seconds ---")
        print(f"Epoch {epoch}, Iteration {iter_counter}")
        print(f"Train Set Loss: {loss_hist[counter]:.2f}")
        print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
        print_batch_accuracy(data, targets, train=True)
        print_batch_accuracy(test_data, test_targets, train=False)
        print("\n")

    # Outer training loop
    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(training_loader)

        # Minibatch training loop
        for data, targets in train_batch:
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            net.train()
            spk_rec, mem_rec = net(data.view(batch_size, -1))

            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Test set
            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(validation_loader))
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)

                # Test set forward pass
                test_spk, test_mem = net(test_data.view(batch_size, -1))

                # Test set loss
                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    test_loss += loss(test_mem[step], test_targets)
                test_loss_hist.append(test_loss.item())

                # Print train/test loss/accuracy
                if counter % 50 == 0:
                    train_printer()
                counter += 1
                iter_counter += 1

    pass


if __name__ == "__main__":
    main()
