"""Training, config, and fault-recovery sweep logic for the memristive SNN studies.

This module owns the *compute* for the publication-scale homeostatic
fault-recovery study and its two diagnostic slices, woven beside
:mod:`mnn_torch.models` / :mod:`mnn_torch.effects` (which own the device
primitives). Nothing here plots inside a core and nothing writes files inside a
core: the reusable pieces (:func:`make_config`, :func:`train_snn`,
:func:`evaluate_snn`, :func:`run_condition`, :func:`gate_homeostasis_sweep`,
:func:`dose_response`, :func:`isolate_collapse`, :func:`precompute_device_params`)
RETURN plain data. Result I/O and the heavy full-scale drivers live in
:func:`main` (argparse ``--gate/--dose/--isolate`` x ``--quick/--full``) and are
routed through :mod:`mnn_torch.paths`, so notebooks and ``python -m
mnn_torch.training`` agree on where grids land regardless of CWD.

``torch``/``torchvision`` are imported *lazily* inside the functions so that
importing :mod:`mnn_torch.training` never drags in the heavy tree.

Science (unchanged from the experiments/ scripts it supersedes)
---------------------------------------------------------------
A fully connected memristive spiking net (MSNN, 784->100->10) is trained on
MNIST with the measured SiOx Poole-Frenkel device prior (Eq. forward-g). We
sweep the device-failure probability ``p`` and the stuck-fault polarity:

  * stuck-on  : saturated/shorted synapses drive their postsynaptic line
                HYPERACTIVE -> exactly what the firing-rate homeostatic
                regulariser (Eq. homeo) suppresses -> recovery SHOULD be large.
  * stuck-off : dead/open synapses go silent -> the homeostat has nothing to act
                on -> the mechanism-NEGATIVE control (little/no recovery).

For every (polarity, p) cell we train two arms -- homeostasis on vs off -- and
report the recovery gap (homeo - no_homeo). ``p = 0`` (faults disabled) is the
clean ceiling and its "recovery" is the generic-regularisation baseline the
fault-driven recovery must rise above. The mechanism test contrasts peak
stuck-on recovery against stuck-off, with bootstrap CIs across seeds.

The production entry (spawn Pool, dataset matrix, MCSNN path) folds in behind
``--full``; the default ``--quick`` runs a tiny in-kernel-safe sweep.
"""
from __future__ import annotations

import os
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from . import paths

warnings.filterwarnings("ignore")

__all__ = [
    "make_config",
    "_make_config",
    "evaluate_snn",
    "train_snn",
    "run_condition",
    "precompute_device_params",
    "gate_homeostasis_sweep",
    "dose_response",
    "isolate_collapse",
    "main",
]


# ==========================================================================
# Config builders
#
# The three source scripts (gate_homeostasis_recovery, dose_response,
# isolate_collapse) each had a near-identical memristive-config builder. They
# are unified here into one ``make_config`` that PRESERVES every key and value
# from all three, with the union of their optional knobs exposed as keyword
# arguments defaulted to each script's original value:
#
#   * ``prob`` -> disturbance_probability (all three).
#   * ``homeo`` -> homeostasis_dropout (all three).
#   * ``nonlinear`` -> PF (True) vs Ohmic (False) forward map. gate/dose used
#     True; isolate toggled it. Default True.
#   * ``disturb_mode`` -> "device_fixed" (gate default), "device" (dose/isolate),
#     or "fixed" (isolate). Default "device_fixed".
#   * ``stuck_polarity`` -> "on"/"off"/"split". gate swept it; dose/isolate never
#     set it (so it stayed the layer default). Default None => key omitted, which
#     reproduces dose/isolate exactly.
#   * ``disturb_conductance``: gate used ``prob > 0`` (p=0 truly disables the
#     path); dose used the explicit ``faulty`` flag; isolate hardwired True.
#     Default None => derive as ``prob > 0`` (the gate rule); pass an explicit
#     bool to reproduce dose (faulty) / isolate (True).
#   * ``homeostasis_r_th`` / ``homeostasis_n`` / ``homeostasis_threshold``:
#     identical across all three (0.8 / 1.0 / 5); kept as defaults.
#   * ``pf_frozen_variability``: gate set it True (fixed-device reading); dose/
#     isolate omitted it (layer default). Default None => omit unless requested.
# ==========================================================================
def make_config(dev_params, *, prob, homeo, nonlinear=True,
                disturb_mode="device_fixed", stuck_polarity=None,
                disturb_conductance=None, pf_frozen_variability=None,
                homeostasis_r_th=0.8, homeostasis_n=1.0, homeostasis_threshold=5):
    """Build a memristive config for a single training run.

    ``dev_params`` is the 8-tuple from :func:`precompute_device_params`
    ``(G_off, G_on, R, c, d_epsilon, pf_slopes, pf_intercepts, pf_covariance)``.

    See the module comment for how every key from the three original builders is
    preserved. When ``disturb_conductance is None`` the disturbance path is
    enabled iff ``prob > 0`` (the gate rule: ``prob == 0`` is a true clean
    ceiling, not the fault sampler run at probability 0).
    """
    G_off, G_on, R, c, d_eps, slopes, intercepts, cov = dev_params
    if disturb_conductance is None:
        disturb_conductance = prob > 0.0
    cfg = {
        "ideal": False,
        "k_V": 0.5,
        "G_off": float(G_off),
        "G_on": float(G_on),
        "R": R,
        "c": c,
        "d_epsilon": d_eps,
        # Poole-Frenkel nonlinear forward map (Eq. forward-g).
        "nonlinear": nonlinear,
        "pf_slopes": slopes,
        "pf_intercepts": intercepts,
        "pf_covariance": cov,
        "pf_stochastic": True,
        # faults (swept).
        "disturb_conductance": disturb_conductance,
        "disturb_mode": disturb_mode,
        "disturbance_probability": float(prob),
        "disturbance_sigma_g": 0.5,
        # homeostasis (sigmoidal firing-rate regulariser, Eq. homeo).
        "homeostasis_dropout": homeo,
        "homeostasis_mode": "regulariser",
        "homeostasis_r_th": homeostasis_r_th,
        "homeostasis_n": homeostasis_n,
        "homeostasis_threshold": homeostasis_threshold,
    }
    # Optional keys only added when set, so dose/isolate configs are reproduced
    # byte-for-byte (they never emitted these keys).
    if stuck_polarity is not None:
        cfg["disturbance_stuck_polarity"] = stuck_polarity
    if pf_frozen_variability is not None:
        cfg["pf_frozen_variability"] = pf_frozen_variability
    return cfg


def _make_config(dev_params, *, prob, homeo, disturb_mode="device_fixed",
                 stuck_polarity="split"):
    """Gate-sweep config builder (compat alias, exact keys of the original
    ``gate_homeostasis_recovery._make_config``).

    Sets ``pf_frozen_variability=True`` (the fixed-device reading, frozen PF
    residual) and threads ``stuck_polarity`` through -- i.e. the gate arm.
    """
    return make_config(dev_params, prob=prob, homeo=homeo, nonlinear=True,
                       disturb_mode=disturb_mode, stuck_polarity=stuck_polarity,
                       disturb_conductance=(prob > 0.0),
                       pf_frozen_variability=True)


# ==========================================================================
# Evaluate / train (shared by every arm)
# ==========================================================================
def evaluate_snn(net, loader, device):
    """Rate-coded top-1 accuracy (%) of ``net`` on ``loader``.

    Prediction = argmax of the summed output spike train (rate code), matching
    the training loss. Flattens each input to ``(B, features)`` for the MSNN path.
    """
    import torch

    net.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.shape[0], -1).to(device)
            y = y.to(device)
            spk, _ = net(x)
            pred = spk.sum(dim=0).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
    return 100.0 * correct / total


def train_snn(cfg, loaders, device, *, seed, epochs, num_steps, lr):
    """Train an MSNN(784->100->10) under ``cfg`` and return ``(final_acc, hist)``.

    ``hist`` is the per-epoch test accuracy list (as dose/isolate returned).
    Rate-coded cross-entropy on the summed output spikes; Adam optimiser.
    """
    import torch
    import torch.nn as nn

    from .models import MSNN

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, test_loader = loaders
    net = MSNN(28 * 28, 100, 10, num_steps=num_steps, beta=0.95,
               memristive_config=cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    hist = []
    for _ in range(epochs):
        net.train()
        for x, y in train_loader:
            x = x.view(x.shape[0], -1).to(device)
            y = y.to(device)
            spk, _ = net(x)
            loss = loss_fn(spk.sum(dim=0), y)  # rate-coded CE on summed spikes
            opt.zero_grad()
            loss.backward()
            opt.step()
        hist.append(evaluate_snn(net, test_loader, device))
    return hist[-1], hist


# ==========================================================================
# Gate-sweep worker (picklable; imports torch lazily)
# ==========================================================================
def run_condition(job):
    """Train + evaluate one gate-sweep cell. Returns the ``job`` dict + ``acc``.

    Picklable entry point for the ProcessPool. ``job`` carries every scalar plus
    the precomputed device-parameter arrays, so no global state crosses the
    process boundary. Lifted verbatim from
    ``gate_homeostasis_recovery.run_condition`` (routed through this module's
    :func:`_make_config`, :func:`mnist_loaders`, :func:`evaluate_snn`).
    """
    import torch
    import torch.nn as nn

    from .data import mnist_loaders
    from .models import MSNN

    if job["threads"]:
        torch.set_num_threads(job["threads"])
    device = torch.device(job["device"])

    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"])

    cfg = _make_config(job["dev_params"], prob=job["prob"], homeo=job["homeo"],
                       disturb_mode=job["disturb_mode"],
                       stuck_polarity=job["stuck_polarity"])
    train_loader, test_loader = mnist_loaders(
        job["data"], seed=job["seed"], batch_size=job["batch_size"],
        train_subset=job["train_subset"], test_subset=job["test_subset"])
    net = MSNN(28 * 28, 100, 10, num_steps=job["num_steps"], beta=0.95,
               memristive_config=cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=job["lr"])
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(job["epochs"]):
        net.train()
        for x, y in train_loader:
            x = x.view(x.shape[0], -1).to(device)
            y = y.to(device)
            spk, _ = net(x)
            loss = loss_fn(spk.sum(dim=0), y)  # rate-coded CE on summed spikes
            opt.zero_grad()
            loss.backward()
            opt.step()
    job = dict(job)
    job["acc"] = evaluate_snn(net, test_loader, device)
    return job


# ==========================================================================
# Device-parameter precompute (fit the SiOx prior ONCE)
# ==========================================================================
def precompute_device_params(data_dir=None):
    """Fit the SiOx device parameters ONCE; return the 8-tuple broadcast to workers.

    Reads the bundled multistate ``.mat`` via :func:`mnn_torch.paths.device_data_mat`
    (``data_dir`` overrides the directory), fits ``G_off/G_on`` and the
    Poole-Frenkel regression, and returns
    ``(G_off, G_on, R, c, d_epsilon, pf_slopes, pf_intercepts, pf_covariance)``.
    """
    from .devices import load_SiOx_multistate
    from .effects import (
        compute_PooleFrenkel_parameters,
        compute_PooleFrenkel_regression_parameters,
    )

    if data_dir is None:
        mat = str(paths.device_data_mat())
    else:
        mat = os.path.join(str(data_dir), "SiO_x-multistate-data.mat")
    exp = load_SiOx_multistate(mat)
    G_off, G_on, R, c, d_eps = compute_PooleFrenkel_parameters(exp)
    slopes, intercepts, cov = compute_PooleFrenkel_regression_parameters(R, c, d_eps)
    return (G_off, G_on, R, c, d_eps, slopes, intercepts, cov)


# ==========================================================================
# Small helpers (git provenance, device pick, bootstrap CI)
# ==========================================================================
def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(paths.data_dir().parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def pick_device(requested):
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _boot_ci(vals, n_boot=10000, seed=0):
    """95% bootstrap CI of the mean (percentile method). Lifted from
    entry_gate_sweep._boot_ci."""
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        v = float(vals.mean()) if len(vals) else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    boot = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ==========================================================================
# Gate-sweep CORE (serial, returns plain data -- for the in-kernel quick run)
# ==========================================================================
def gate_homeostasis_sweep(*, dev_params=None, data_dir=None, seeds=(0,),
                           probs=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                           polarities=("on", "off"), disturb_mode="device_fixed",
                           epochs=4, num_steps=8, train_subset=6000,
                           test_subset=2000, batch_size=64, lr=5e-4,
                           device="cpu", threads=0, workers=1, verbose=True):
    """Run the homeostatic fault-recovery sweep and RETURN the result grid dict.

    Serial by default (``workers=1``) so it is safe to call in a notebook kernel;
    ``workers>1`` fans the independent (seed x polarity x prob x arm) cells out
    over a :class:`ProcessPoolExecutor` of :func:`run_condition` workers.

    Returns a dict with the raw accuracy grids ``acc[f"{pol}|{arm}"] -> (n_seeds,
    n_probs)``, the per-cell recovery + bootstrap CIs, the fault-peak per
    polarity, and (when both ``on`` and ``off`` are swept) the mechanism-test
    contrast. No file I/O, no plotting -- :func:`main` handles persistence.
    """
    seeds = list(seeds)
    probs = list(probs)
    polarities = list(polarities)
    if dev_params is None:
        dev_params = precompute_device_params(data_dir)
    data = str(paths.data_dir()) if data_dir is None else str(data_dir)

    # ---- build the independent job list ----
    jobs = []
    for si, seed in enumerate(seeds):
        for pol in polarities:
            for pi, prob in enumerate(probs):
                for homeo in (False, True):
                    jobs.append(dict(
                        si=si, seed=seed, pi=pi, prob=prob, polarity=pol,
                        stuck_polarity=pol, homeo=homeo,
                        disturb_mode=disturb_mode,
                        epochs=epochs, num_steps=num_steps,
                        train_subset=train_subset, test_subset=test_subset,
                        batch_size=batch_size, lr=lr, data=data,
                        device=device, threads=threads,
                        dev_params=dev_params,
                    ))
    if verbose:
        print(f"{len(jobs)} independent conditions "
              f"({len(seeds)} seeds x {len(polarities)} polarities x "
              f"{len(probs)} probs x 2 arms)")

    # results[(polarity, 'homeo'|'no')] -> (n_seeds, n_probs)
    acc = {}
    for pol in polarities:
        acc[(pol, "homeo")] = np.zeros((len(seeds), len(probs)))
        acc[(pol, "no")] = np.zeros((len(seeds), len(probs)))

    def store(done):
        arm = "homeo" if done["homeo"] else "no"
        acc[(done["polarity"], arm)][done["si"], done["pi"]] = done["acc"]
        if verbose:
            print(f"  done: seed{done['seed']} {done['polarity']:5s} "
                  f"p={done['prob']:.2f} {'homeo ' if done['homeo'] else 'nohomeo'} "
                  f"-> {done['acc']:.2f}%")

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_condition, j) for j in jobs]
            for fut in as_completed(futs):
                store(fut.result())
    else:
        for j in jobs:
            store(run_condition(j))

    # ---- summarise (recovery, per-cell CIs, fault-peak, mechanism test) ----
    probs_arr = np.asarray(probs, dtype=float)
    fault_j = [j for j, p in enumerate(probs) if p > 0.0]

    summary = {}
    for pol in polarities:
        nh, h = acc[(pol, "no")], acc[(pol, "homeo")]
        rec = h - nh
        cell = []
        for j, p in enumerate(probs):
            lo, hi = _boot_ci(rec[:, j], seed=j)
            cell.append(dict(p=p, no_homeo=float(nh[:, j].mean()),
                             homeo=float(h[:, j].mean()),
                             recovery=float(rec[:, j].mean()), ci_lo=lo, ci_hi=hi))
        peak = None
        if fault_j:
            jpk = max(fault_j, key=lambda j: rec[:, j].mean())
            plo, phi = _boot_ci(rec[:, jpk], seed=jpk)
            peak = dict(p=probs[jpk], recovery=float(rec[:, jpk].mean()),
                        ci_lo=plo, ci_hi=phi)
        base = float((h - nh)[:, 0].mean()) if probs_arr[0] == 0.0 else float("nan")
        summary[pol] = dict(cells=cell, peak=peak, ideal_effect=base)

    mechanism = None
    if "on" in polarities and "off" in polarities and fault_j:
        rec_on = acc[("on", "homeo")] - acc[("on", "no")]
        rec_off = acc[("off", "homeo")] - acc[("off", "no")]
        j_on = max(fault_j, key=lambda j: rec_on[:, j].mean())
        diff = rec_on[:, j_on] - rec_off[:, j_on]   # paired by seed
        dlo, dhi = _boot_ci(diff, seed=99)
        if dlo > 0:
            verdict = "MECHANISM-SPECIFIC (on > off, CI excludes 0)"
        elif dhi < 0:
            verdict = "REVERSED (off > on, CI excludes 0) -- unexpected"
        else:
            verdict = "NOT separable (CI crosses 0; need more seeds)"
        mechanism = dict(p=probs[j_on], diff=float(diff.mean()),
                         ci_lo=dlo, ci_hi=dhi, verdict=verdict)

    return {
        "probs": probs, "polarities": polarities, "seeds": len(seeds),
        "seed_values": seeds, "epochs": epochs, "num_steps": num_steps,
        "disturb_mode": disturb_mode, "train_subset": train_subset,
        "test_subset": test_subset, "dataset": "mnist",
        "acc": {f"{pol}|{arm}": acc[(pol, arm)]
                for pol in polarities for arm in ("no", "homeo")},
        "summary": summary, "mechanism_test": mechanism,
    }


def _print_gate_summary(grid):
    """Human-readable dump of a :func:`gate_homeostasis_sweep` grid."""
    probs = grid["probs"]
    print(f"\n================ SWEEP SUMMARY (mean over {grid['seeds']} seed[s], "
          "95% bootstrap CI on recovery) ================")
    acc = grid["acc"]
    for pol in grid["polarities"]:
        nh = acc[f"{pol}|no"]
        h = acc[f"{pol}|homeo"]
        rec = (h - nh).mean(0)
        s = grid["summary"][pol]
        print(f"\n  --- stuck-{pol} ---")
        print(f"  {'p':>5}  {'no-homeo':>9}  {'homeo':>9}  {'recovery':>9}  {'95% CI':>16}")
        for cell in s["cells"]:
            print(f"  {cell['p']:5.2f}  {cell['no_homeo']:8.2f}%  {cell['homeo']:8.2f}%  "
                  f"{cell['recovery']:+8.2f}   [{cell['ci_lo']:+.2f},{cell['ci_hi']:+.2f}]")
        pk = s["peak"]
        if pk is not None:
            print(f"  ideal effect (p=0) = {s['ideal_effect']:+.2f} pp ; "
                  f"peak fault recovery = {pk['recovery']:+.2f} pp "
                  f"[{pk['ci_lo']:+.2f},{pk['ci_hi']:+.2f}] at p={pk['p']:.2f}")
    mt = grid["mechanism_test"]
    if mt is not None:
        print(f"\n  MECHANISM TEST @ p={mt['p']:.2f}: recovery(on)-recovery(off) = "
              f"{mt['diff']:+.2f} pp [{mt['ci_lo']:+.2f},{mt['ci_hi']:+.2f}] -> {mt['verdict']}")


# ==========================================================================
# Dose-response CORE (single fault rate; returns plain data)
# ==========================================================================
def dose_response(rate, *, dev_params=None, data_dir=None, seed=0, epochs=5,
                  num_steps=5, train_subset=4000, test_subset=1500,
                  batch_size=64, lr=5e-4, device=None, verbose=True):
    """Dose-response slice at ONE device-failure ``rate``; RETURN the result dict.

    ``rate == 0`` produces the fault-rate-independent reference conditions
    (``ideal`` / ``ideal+homeo``); ``rate > 0`` measures ``faulty`` /
    ``faulty+homeo`` and their ``recovery`` gap. Uses ``disturb_mode="device"``
    (the dose-response setting) and the Poole-Frenkel forward map. No file I/O.
    """
    import torch

    if device is None:
        device = torch.device(pick_device("auto"))
    elif isinstance(device, str):
        device = torch.device(device)
    if dev_params is None:
        dev_params = precompute_device_params(data_dir)

    from .data import ensure_mnist, mnist_loaders
    data = ensure_mnist(data_dir)
    loaders = mnist_loaders(data, seed=seed, batch_size=batch_size,
                            train_subset=train_subset, test_subset=test_subset)

    out = {"rate": rate, "device": str(device), "epochs": epochs,
           "num_steps": num_steps, "train_subset": train_subset, "seed": seed}

    if rate == 0.0:
        arms = [("ideal", False), ("ideal+homeo", True)]
        faulty = False
    else:
        arms = [("faulty", False), ("faulty+homeo", True)]
        faulty = True
    for name, homeo in arms:
        cfg = make_config(dev_params, prob=rate, homeo=homeo, nonlinear=True,
                          disturb_mode="device", disturb_conductance=faulty)
        acc, hist = train_snn(cfg, loaders, device, seed=seed, epochs=epochs,
                              num_steps=num_steps, lr=lr)
        out[name] = acc
        out[name + "_hist"] = hist
        if verbose:
            print(f"[rate={rate}] {name:14s} {acc:.2f}%  "
                  f"hist={['%.1f' % a for a in hist]}", flush=True)
    if rate != 0.0:
        out["recovery"] = out["faulty+homeo"] - out["faulty"]
        if verbose:
            print(f"[rate={rate}] recovery = {out['recovery']:+.2f}%", flush=True)
    return out


# ==========================================================================
# Isolate-collapse CORE (one (map, mode) combination; returns plain data)
# ==========================================================================
def isolate_collapse(map, mode, *, dev_params=None, data_dir=None, rate=0.5,
                     seed=0, epochs=5, num_steps=5, train_subset=4000,
                     test_subset=1500, batch_size=64, lr=5e-4, device="cpu",
                     verbose=True):
    """Isolate the faulty-baseline collapse: one ``(map, mode)`` combination.

    ``map`` in {"pf", "ohmic"} toggles the forward map (Poole-Frenkel vs Ohmic
    ``I=V*G``); ``mode`` in {"device", "fixed"} toggles the disturbance sampler.
    Measures ``faulty`` / ``faulty+homeo`` at fixed ``rate`` and their
    ``recovery``. RETURNS the result dict (no file I/O). CPU by default, matching
    the parallel-on-CPU dose sweep.
    """
    import torch

    device = torch.device(device) if isinstance(device, str) else device
    nonlinear = map == "pf"
    if dev_params is None:
        dev_params = precompute_device_params(data_dir)

    from .data import ensure_mnist, mnist_loaders
    data = ensure_mnist(data_dir)
    loaders = mnist_loaders(data, seed=seed, batch_size=batch_size,
                            train_subset=train_subset, test_subset=test_subset)

    out = {"map": map, "mode": mode, "rate": rate, "epochs": epochs,
           "num_steps": num_steps, "train_subset": train_subset, "seed": seed}
    for name, homeo in [("faulty", False), ("faulty+homeo", True)]:
        cfg = make_config(dev_params, prob=rate, homeo=homeo, nonlinear=nonlinear,
                          disturb_mode=mode, disturb_conductance=True)
        acc, hist = train_snn(cfg, loaders, device, seed=seed, epochs=epochs,
                              num_steps=num_steps, lr=lr)
        out[name] = acc
        out[name + "_hist"] = hist
        if verbose:
            print(f"[{map}/{mode}] {name:14s} {acc:.2f}%  "
                  f"hist={['%.1f' % a for a in hist]}", flush=True)
    out["recovery"] = out["faulty+homeo"] - out["faulty"]
    if verbose:
        print(f"[{map}/{mode}] recovery = {out['recovery']:+.2f}%", flush=True)
    return out


# ==========================================================================
# CLI drivers (argparse dispatch; result I/O routed through paths)
# ==========================================================================
def _gate_main(args):
    """Gate-sweep driver: quick in-kernel sweep, or the production spawn-Pool
    dataset-matrix sweep behind ``--full``. Writes the grid + summary via paths."""
    import json

    if args.full:
        # Production entry: the entry_gate_sweep spawn-Pool / dataset-matrix /
        # MCSNN path lives in a dedicated function so the heavy imports (spawn
        # multiprocessing) stay out of the quick path.
        return _gate_full(args)

    # ---- quick, in-kernel-safe sweep (serial by default) ----
    workers = args.workers if args.workers and args.workers > 1 else 1
    device = "cpu" if workers > 1 else pick_device(args.device)

    from .data import ensure_mnist
    data = ensure_mnist()  # download MNIST once into paths.data_dir()
    dev_params = precompute_device_params()
    print(f"[gate] SiOx fit: G_off={dev_params[0]:.3e} G_on={dev_params[1]:.3e} "
          f"slopes={np.round(dev_params[5], 3)}")

    # Deliberately tiny for a fast smoke run: few epochs, small subset, coarse
    # prob grid. Scale up with --full for the publication sweep.
    grid = gate_homeostasis_sweep(
        dev_params=dev_params, data_dir=data, seeds=(0,),
        probs=(0.0, 0.2, 0.4), polarities=("on", "off"),
        disturb_mode="device_fixed", epochs=1, num_steps=5,
        train_subset=512, test_subset=512, batch_size=64, lr=5e-4,
        device=device, threads=(2 if workers > 1 else 0), workers=workers)
    _print_gate_summary(grid)

    paths.save_result("gate_sweep_results.npy", grid)
    # summary JSON + CSV mirror the production schema (under paths.results_dir()).
    _write_gate_outputs(grid, git_commit())
    print(f"\n[gate] wrote gate_sweep_results.npy + gate_sweep_summary.json + "
          f"gate_sweep_results.csv to {paths.results_dir()}")
    return grid


def _write_gate_outputs(grid, commit):
    """Persist a gate grid to CSV + JSON + .npy under ``paths.results_dir()``."""
    import csv
    import json

    rdir = paths.results_dir()
    # summary JSON (per-polarity cells + mechanism test), production schema.
    summary = {}
    for pol in grid["polarities"]:
        summary[pol] = grid["summary"][pol]["cells"]
    if grid["mechanism_test"] is not None:
        mt = grid["mechanism_test"]
        summary["_mechanism_test"] = dict(p=mt["p"], diff=mt["diff"],
                                          ci_lo=mt["ci_lo"], ci_hi=mt["ci_hi"],
                                          verdict=mt["verdict"])
    with open(rdir / "gate_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # long-format CSV (one row per polarity x seed x prob), provenance columns.
    acc = grid["acc"]
    seeds = grid["seed_values"]
    rows = []
    for pol in grid["polarities"]:
        nh_grid = acc[f"{pol}|no"]
        h_grid = acc[f"{pol}|homeo"]
        for si, seed in enumerate(seeds):
            for pi, prob in enumerate(grid["probs"]):
                nh = nh_grid[si, pi]
                h = h_grid[si, pi]
                rows.append([commit, seed, grid["epochs"], grid["num_steps"],
                             grid["train_subset"], grid["disturb_mode"], pol,
                             f"{prob:.3f}", f"{nh:.2f}", f"{h:.2f}", f"{h - nh:.2f}"])
    out_csv = rdir / "gate_sweep_results.csv"
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["commit", "seed", "epochs", "num_steps", "train_subset",
                        "disturb_mode", "stuck_polarity", "prob",
                        "no_homeo", "homeo", "recovery"])
        w.writerows(rows)


def _gate_full(args):
    """Production gate sweep: spawn-Pool, multi-seed, publication grid.

    Folds in the entry_gate_sweep fan-out (many seeds x fault grid x polarity x
    arm, bootstrap CIs, mechanism test) using this module's picklable
    :func:`run_condition` worker over a ProcessPool. Writes gate_sweep.npy +
    gate_sweep_summary.json + gate_sweep_results.csv via paths.
    """
    workers = args.workers if args.workers and args.workers > 0 else max(
        1, (os.cpu_count() or 4) - 2)
    device = "cpu" if workers > 1 else pick_device(args.device)

    from .data import ensure_mnist
    data = ensure_mnist()
    dev_params = precompute_device_params()
    print(f"[gate] SiOx fit: G_off={dev_params[0]:.3e} G_on={dev_params[1]:.3e} "
          f"slopes={np.round(dev_params[5], 3)}")
    print(f"[gate] FULL sweep: seeds={args.seeds} probs={args.probs} "
          f"polarity={args.polarity} workers={workers} device={device}")

    grid = gate_homeostasis_sweep(
        dev_params=dev_params, data_dir=data, seeds=tuple(range(args.seeds)),
        probs=tuple(args.probs), polarities=tuple(args.polarity),
        disturb_mode=args.disturb_mode, epochs=args.epochs,
        num_steps=args.num_steps, train_subset=args.train_subset,
        test_subset=args.test_subset, batch_size=args.batch_size, lr=args.lr,
        device=device, threads=(args.threads if workers > 1 else 0),
        workers=workers)
    _print_gate_summary(grid)
    paths.save_result("gate_sweep.npy", grid)
    _write_gate_outputs(grid, git_commit())
    print(f"\n[gate] wrote gate_sweep.npy + gate_sweep_summary.json + "
          f"gate_sweep_results.csv to {paths.results_dir()}")
    return grid


def _dose_main(args):
    """Dose-response driver: one rate per invocation, JSON per rate under
    data/results/dose/<rate>.json."""
    import json

    from .data import ensure_mnist
    ensure_mnist()
    dev_params = precompute_device_params()

    # --quick shrinks the training budget; --full uses the original defaults.
    if args.quick and not args.full:
        kw = dict(epochs=1, num_steps=5, train_subset=512, test_subset=512)
    else:
        kw = dict(epochs=5, num_steps=5, train_subset=4000, test_subset=1500)
    out = dose_response(args.rate, dev_params=dev_params, seed=args.seed,
                        device=("cpu" if args.cpu else None), **kw)

    dose_dir = paths.results_dir() / "dose"
    dose_dir.mkdir(parents=True, exist_ok=True)
    fn = dose_dir / f"rate_{args.rate:.2f}.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {fn}", flush=True)
    return out


def _isolate_main(args):
    """Isolate-collapse driver: one (map, mode) per invocation, JSON under
    data/results/isolate/<map>_<mode>.json."""
    import json

    from .data import ensure_mnist
    ensure_mnist()
    dev_params = precompute_device_params()

    if args.quick and not args.full:
        kw = dict(epochs=1, num_steps=5, train_subset=512, test_subset=512)
    else:
        kw = dict(epochs=5, num_steps=5, train_subset=4000, test_subset=1500)
    out = isolate_collapse(args.map, args.mode, dev_params=dev_params,
                           rate=args.rate, seed=args.seed, device="cpu", **kw)

    iso_dir = paths.results_dir() / "isolate"
    iso_dir.mkdir(parents=True, exist_ok=True)
    fn = iso_dir / f"{args.map}_{args.mode}.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {fn}", flush=True)
    return out


def main(argv=None):
    """Full-scale reproduction CLI for the training/homeostasis studies.

    ``python -m mnn_torch.training --gate  [--quick|--full] [--workers N]``
    ``python -m mnn_torch.training --dose  --rate R [--quick|--full]``
    ``python -m mnn_torch.training --isolate --map pf --mode device [--quick|--full]``

    ``--quick`` runs a tiny in-kernel-safe version; ``--full`` runs the published
    scale (spawn Pool / multi-seed for --gate). Results are written under
    ``paths.results_dir()`` regardless of CWD.
    """
    import argparse

    ap = argparse.ArgumentParser(description="mnn-torch training / homeostasis studies")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--gate", action="store_true",
                     help="homeostatic fault-recovery sweep (default)")
    grp.add_argument("--dose", action="store_true",
                     help="dose-response slice at a single --rate")
    grp.add_argument("--isolate", action="store_true",
                     help="isolate the collapse: one --map/--mode combination")

    ap.add_argument("--quick", action="store_true",
                    help="tiny in-kernel-safe run (default if neither --quick/--full)")
    ap.add_argument("--full", action="store_true",
                    help="published-scale run (spawn Pool / multi-seed for --gate)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes (gate; 1 = serial/in-kernel)")
    ap.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")

    # gate (full) grid knobs -- mirror entry_gate_sweep defaults.
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--probs", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3])
    ap.add_argument("--polarity", nargs="+", default=["on", "off"],
                    choices=["on", "off", "split"])
    ap.add_argument("--disturb-mode", default="device_fixed",
                    choices=["device_fixed", "device", "fixed"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--train-subset", type=int, default=60000)
    ap.add_argument("--test-subset", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--threads", type=int, default=2,
                    help="torch threads per worker (gate CPU pool path)")

    # dose / isolate knobs.
    ap.add_argument("--rate", type=float, default=0.5,
                    help="device-failure probability (dose: required; isolate: fixed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="dose: force CPU")
    ap.add_argument("--map", choices=["pf", "ohmic"], default="pf",
                    help="isolate: forward map")
    ap.add_argument("--mode", choices=["device", "fixed"], default="device",
                    help="isolate: disturbance sampler")

    args = ap.parse_args(argv)

    if args.dose:
        return _dose_main(args)
    if args.isolate:
        return _isolate_main(args)
    # default -> gate
    return _gate_main(args)


if __name__ == "__main__":
    main()
