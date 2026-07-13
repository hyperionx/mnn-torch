"""Regression tests for publication-run device and loader plumbing."""

from __future__ import annotations

import inspect

import torch

from mnn_torch.data import mnist_loaders
from mnn_torch.training import _move_to_device, pick_device


def test_auto_device_prefers_available_accelerator() -> None:
    selected = pick_device("auto")
    if torch.cuda.is_available():
        assert selected == "cuda"
    elif torch.backends.mps.is_available():
        assert selected == "mps"
    else:
        assert selected == "cpu"


def test_explicit_device_is_preserved() -> None:
    assert pick_device("cpu") == "cpu"


def test_mnist_loader_exposes_nonbreaking_worker_controls() -> None:
    parameters = inspect.signature(mnist_loaders).parameters
    assert parameters["num_workers"].default == 0
    assert parameters["pin_memory"].default is None
    assert parameters["persistent_workers"].default is False


def test_cpu_transfer_helper_preserves_values() -> None:
    source = torch.arange(5)
    moved = _move_to_device(source, torch.device("cpu"))
    assert torch.equal(source, moved)

