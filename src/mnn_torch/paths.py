"""Filesystem paths for the bundled device fixture and result grids.

The experiments in this package are reproduced by notebooks under ``experiments/``
that hold no code themselves: they call the ``run_*`` helpers in the library and read
or write result grids under the top-level ``data/`` directory. This module is the one
place that resolves ``data/`` so that both the notebooks and the module ``main()``
entry points (``python -m mnn_torch.training --full``) agree on where results live,
regardless of the current working directory.

Because the package is installed editable, ``__file__`` resolves inside the real
repository ``src/`` tree, so the repository root -- and its sibling ``data/`` -- is
discoverable from the module location. Set ``MNN_DATA_DIR`` to override (e.g. for a
wheel install or CI where ``data/`` lives elsewhere).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

__all__ = [
    "data_dir",
    "results_dir",
    "device_data_mat",
    "save_result",
    "load_result",
]


def data_dir() -> Path:
    """Absolute path to the package ``data/`` directory (created if absent).

    Honours the ``MNN_DATA_DIR`` environment variable; otherwise resolves to
    ``<repo-root>/data`` from this module's location (editable install).
    """
    env = os.environ.get("MNN_DATA_DIR")
    if env:
        d = Path(env).expanduser().resolve()
    else:
        # paths.py -> mnn_torch -> src -> <repo-root>
        d = Path(__file__).resolve().parents[2] / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir() -> Path:
    """``data/results`` -- the temporal/store-recall/homeostasis grids the notebooks replay."""
    p = data_dir() / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def device_data_mat() -> Path:
    """Path to the bundled SiOx multistate device fixture ``SiO_x-multistate-data.mat``.

    Resolves to ``data_dir()/SiO_x-multistate-data.mat`` when that exists. When
    ``MNN_DATA_DIR`` relocates the *writable* data dir (for MNIST downloads and
    result grids) the shipped fixture is not there, so we fall back to the
    repository-relative ``<repo-root>/data`` (discoverable from this module's
    location under the editable ``src/`` tree). The fixture is read-only, so it
    always ships with the repo even when results live elsewhere.
    """
    primary = data_dir() / "SiO_x-multistate-data.mat"
    if primary.exists():
        return primary
    repo_data = Path(__file__).resolve().parents[2] / "data" / "SiO_x-multistate-data.mat"
    if repo_data.exists():
        return repo_data
    return primary


def save_result(name: str, obj) -> Path:
    """Pickle-save ``obj`` to ``data/results/<name>`` (``.npy``), returning the path."""
    path = results_dir() / name
    np.save(path, obj, allow_pickle=True)
    return path


def load_result(name: str):
    """Load a result grid written by :func:`save_result` (``allow_pickle`` dict)."""
    return np.load(results_dir() / name, allow_pickle=True).item()
