"""Dataset loading and preparation helpers for the mnn-torch experiments.

Two dataset paths feed the memristive spiking-network studies:

* **MNIST** -- the fault-recovery / homeostasis sweeps.
  :func:`ensure_mnist` downloads it once (into :func:`mnn_torch.paths.data_dir`
  by default) so that pooled worker processes can open it with ``download=False``;
  :func:`mnist_loaders` builds seeded train/test ``DataLoader`` pairs, optionally
  subsetting for a fast smoke run.

* **BCI Competition IV-2a** (BNCI2014_001, 4-class motor imagery) -- the
  biosignal-classification study. :func:`prep_bci2a` downloads and
  epochs it via MOABB/MNE and writes a shippable ``.npz`` so the heavy EEG
  dependency tree never has to enter the training container.

``torch``/``torchvision`` and ``moabb``/``mne`` are imported *lazily* inside the
functions so that importing :mod:`mnn_torch.data` (and hence :mod:`mnn_torch`)
never drags in those heavy trees; only the caller who actually needs a loader
pays for the import.
"""
from __future__ import annotations

import os

import numpy as np

from . import paths

__all__ = [
    "mnist_loaders",
    "ensure_mnist",
    "prep_bci2a",
    # BCI IV-2a preprocessing constants (module-level so callers/tests can read them)
    "FMIN",
    "FMAX",
    "TMIN",
    "TMAX",
    "N_EEG",
    "DECIMATE",
]


# --------------------------------------------------------------------------
# MNIST (fault-recovery / homeostasis sweeps)
# --------------------------------------------------------------------------
def ensure_mnist(data_dir=None):
    """Download MNIST once (train + test) so workers can use ``download=False``.

    ``data_dir`` defaults to :func:`mnn_torch.paths.data_dir`. Returns the
    resolved data directory as a string so callers can thread it to workers.
    """
    import torchvision
    import torchvision.transforms as T

    data_dir = str(paths.data_dir()) if data_dir is None else str(data_dir)
    tfm = T.Compose([T.ToTensor()])
    torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
    return data_dir


def mnist_loaders(data_dir=None, seed=0, batch_size=64, train_subset=None,
                  test_subset=None):
    """Build seeded MNIST train/test ``DataLoader`` pairs.

    Lifted from ``gate_homeostasis_recovery._data_loaders``: the train loader is
    shuffled under a per-run ``torch.Generator`` seed and drops the last partial
    batch; the test loader is deterministic. ``train_subset`` / ``test_subset``
    (if given) take the leading ``n`` examples for a fast smoke run. Assumes
    MNIST is already present (see :func:`ensure_mnist`); opens with
    ``download=False``.
    """
    import torch
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Subset

    data_dir = str(paths.data_dir()) if data_dir is None else str(data_dir)
    tfm = T.Compose([T.ToTensor(), T.Normalize((0,), (1,))])
    train = torchvision.datasets.MNIST(data_dir, train=True, download=False, transform=tfm)
    test = torchvision.datasets.MNIST(data_dir, train=False, download=False, transform=tfm)
    if train_subset:
        train = Subset(train, range(train_subset))
    if test_subset:
        test = Subset(test, range(test_subset))
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              drop_last=True, generator=g)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                             drop_last=True)
    return train_loader, test_loader


# --------------------------------------------------------------------------
# BCI Competition IV-2a (biosignal classification)
# --------------------------------------------------------------------------
# Standard 4-class motor-imagery preprocessing (see prep_bci2a docstring).
FMIN, FMAX = 8.0, 30.0          # mu + beta motor-imagery band
TMIN, TMAX = 0.5, 2.5           # seconds post-cue (avoid the cue transient)
N_EEG = 22                      # drop the 3 EOG channels
DECIMATE = 4                    # 250 Hz -> 62.5 Hz: the band is <=30 Hz so this is
                                # safely above Nyquist, and cuts the flatten dim from
                                # 22x501=11022 to 22x125=2750 (MNIST-scale, no bottleneck).


def _load_bci2a_epochs():
    """Download (cached by MOABB) and epoch BCI IV-2a. Returns X[n,ch,t], y[n] int 0-3,
    subj[n] int, and the class-name list. MOABB/MNE imported lazily."""
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    # 4-class MI, our band-pass and window; MOABB returns z-scoring-free volts.
    paradigm = MotorImagery(n_classes=4, fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX,
                            channels=None, resample=None)
    ds = BNCI2014_001()
    X, labels, meta = paradigm.get_data(dataset=ds)        # X: (n, n_chan, n_time)
    X = X[:, :N_EEG, :].astype(np.float32)                 # EEG channels only
    if DECIMATE > 1:                                       # anti-alias-safe: band <= 30 Hz
        X = X[:, :, ::DECIMATE]                            # 501 -> ~125 timepoints

    classes = sorted(np.unique(labels).tolist())           # e.g. ['feet','left_hand',...]
    cls_to_int = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_to_int[l] for l in labels], dtype=np.int64)
    subj = meta["subject"].to_numpy().astype(np.int64)

    # Per-trial, per-channel z-score (the SNN front-end expects standardised input, and
    # this matches the biosignal.py convention in the sibling repo).
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True) + 1e-7
    X = (X - mu) / sd
    return X, y, subj, classes


def _csp_lda_sanity(X, y, subj, seed=0):
    """Within-subject CSP+LDA 5-fold accuracy, averaged over subjects -- the clean-ceiling
    gate. Uses MNE's CSP on the covariance of the band-passed epochs."""
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    accs = []
    for s in np.unique(subj):
        m = subj == s
        Xs, ys = X[m], y[m]
        clf = Pipeline([("csp", CSP(n_components=8, reg="ledoit_wolf", log=True)),
                        ("lda", LinearDiscriminantAnalysis())])
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        a = cross_val_score(clf, Xs, ys, cv=skf, scoring="accuracy").mean()
        accs.append(a)
        print(f"  subject {s:2d}: CSP+LDA 4-class acc = {a:.3f}  (n={m.sum()})")
    return float(np.mean(accs)), float(np.std(accs))


def prep_bci2a(out=None, skip_sanity=False):
    """Pre-cache the BCI Competition IV-2a (BNCI2014_001) 4-class motor-imagery EEG
    as a shippable ``.npz``.

    Downloads + epochs the dataset via MOABB, optionally runs the within-subject
    CSP+LDA clean-ceiling sanity decode, and writes ``X/y/subj/classes`` plus the
    band-pass/window metadata to ``out`` (default:
    ``paths.data_dir()/"bci2a_epochs.npz"``). Returns the absolute output path.

    MOABB/MNE are heavy and would re-download on every batch run, so the download +
    epoching happens ONCE here (in an environment that already has moabb/mne) and
    only the dense array is shipped -- exactly as the SiOx ``.mat`` is shipped.
    """
    out = paths.data_dir() / "bci2a_epochs.npz" if out is None else out
    out = os.path.abspath(str(out))

    print("[prep] downloading + epoching BCI IV-2a (BNCI2014_001) via MOABB ...")
    X, y, subj, classes = _load_bci2a_epochs()
    print(f"[prep] epochs: X={X.shape} (trials, ch, time)  classes={classes}  "
          f"subjects={np.unique(subj).tolist()}")
    print(f"[prep] class balance: {np.bincount(y).tolist()}")

    mean_acc = std_acc = float("nan")
    if not skip_sanity:
        print("[prep] CSP+LDA within-subject sanity decode (clean-ceiling gate) ...")
        mean_acc, std_acc = _csp_lda_sanity(X, y, subj)
        print(f"[prep] CSP+LDA mean 4-class accuracy = {mean_acc:.3f} +/- {std_acc:.3f} "
              f"(chance 0.25; published CSP+LDA ~0.67)")
        if mean_acc < 0.40:
            print("[prep] WARNING: clean ceiling is low -- check band-pass/window before "
                  "spending SNN effort.")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, X=X, y=y, subj=subj, classes=np.array(classes),
                        fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX,
                        csp_lda_acc=mean_acc, csp_lda_std=std_acc,
                        sfreq=250.0, n_eeg=N_EEG)
    print(f"[prep] wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    return out


def main(argv=None):
    """CLI wrapper for :func:`prep_bci2a` (mirrors the old prep_bci2a.py entry).

    ``python -m mnn_torch.data [--out PATH] [--skip-sanity]``
    """
    import argparse

    ap = argparse.ArgumentParser(description="Pre-cache BCI IV-2a epochs as a .npz")
    ap.add_argument("--out", default=None,
                    help="output .npz path (default: paths.data_dir()/bci2a_epochs.npz)")
    ap.add_argument("--skip-sanity", action="store_true")
    a = ap.parse_args(argv)
    prep_bci2a(out=a.out, skip_sanity=a.skip_sanity)


if __name__ == "__main__":
    main()
