import time
import numpy as np
import torch, torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

from mnn_torch.devices import load_SiOx_multistate
from mnn_torch.models import MSNN, MSCNN
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

def main():
    start_time = time.time()
    torch.set_printoptions(threshold=10)
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

    training_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    validation_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Network Architecture
    num_inputs = 28 * 28
    num_hidden = 25
    num_outputs = 10

    # Temporal Dynamics
    num_steps = 25
    beta = 0.95

    # Memrisitive Configuration
    PF_config = {
                 "experimental_data": experimental_data,
                 "k_V": 0.5,
                 "ideal": False,
                 "disturb_conductance": True,
                 }


    net = MSNN(
        device,
        num_inputs,
        num_hidden,
        num_outputs,
        num_steps,
        beta,
        memrisitive_config=PF_config,
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

def main2():
    # dataloader arguments
    batch_size = 128
    data_path = "C:\\Users\\Mr_VC\\git\\mnn-torch\\data"
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

    training_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    validation_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=True, drop_last=True
    )

    spike_grad = surrogate.fast_sigmoid(slope=25)
    beta = 0.5
    num_steps = 50

    # Network Architecture
    num_kernels = 5
    num_conv1 = 12
    num_conv2 = 64
    max_pooling = 2
    num_hidden = num_conv2 * 4 * 4
    num_outputs = 10

    memrisitive_config = {
                    # "experimental_data": experimental_data,
                    # "k_V": 0.5,
                    "ideal": True,
                    "disturb_conductance": False,
                    }

    net = MSCNN(device=device, beta=beta, spike_grad=spike_grad, batch_size=batch_size,
                num_kernels=num_kernels, num_conv1=num_conv1, num_conv2=num_conv2,
                max_pooling=max_pooling, num_hidden=num_hidden, num_outputs=num_outputs,
                memrisitive_config=memrisitive_config)

    loss_fn = SF.ce_rate_loss()

    def forward_pass(net, num_steps, data):
        mem_rec = []
        spk_rec = []
        utils.reset(net)  # resets hidden states for all LIF neurons in net

        for step in range(num_steps):
            spk_out, mem_out = net(data)
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)
    
    def batch_accuracy(train_loader, net, num_steps):
        with torch.no_grad():
            total = 0
            acc = 0
            net.eval()

            train_loader = iter(train_loader)
            for data, targets in train_loader:
                data = data.to(device)
                targets = targets.to(device)
                spk_rec, _ = forward_pass(net, num_steps, data)

                acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
                total += spk_rec.size(1)

        return acc/total
    

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
    num_epochs = 1
    loss_hist = []
    test_acc_hist = []
    counter = 0

    # Outer training loop
    for epoch in range(num_epochs):

        # Training loop
        for data, targets in iter(training_loader):
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            net.train()
            spk_rec, _ = forward_pass(net, num_steps, data)

            # initialize the loss & sum over time
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Test set
            if counter % 50 == 0:
                with torch.no_grad():
                    net.eval()

                    # Test set forward pass
                    test_acc = batch_accuracy(validation_loader, net, num_steps)
                    print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
                    test_acc_hist.append(test_acc.item())

            counter += 1

if __name__ == "__main__":
    main2()
