import json
import hashlib
import re
from pathlib import Path

import nbformat
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_cuda_first_torch_resolution_is_explicit():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert '"torch==2.12.1"' in pyproject
    assert '"torchvision==0.27.1"' in pyproject
    assert 'url = "https://download.pytorch.org/whl/cu130"' in pyproject
    assert 'index = "pytorch-cu130"' in pyproject


def test_figure_manifest_contract():
    manifest = json.loads((ROOT / "experiments" / "figure_manifest.json").read_text())
    figures = manifest["figures"]
    required = {
        "filename", "tier", "notebook", "profile", "device", "seeds",
        "data_hash", "reference_hash", "provenance_class", "claim_status",
    }
    assert manifest["figure_count"] == len(figures) == 17
    assert manifest["active_manuscript_count"] == sum(x["tier"] == "main" for x in figures) == 8
    assert len({x["filename"] for x in figures}) == 17
    assert all(set(x) == required for x in figures)
    assert {x["provenance_class"] for x in figures} <= set(manifest["provenance_classes"])
    assert {"snn_schematic.png", "homeostatic_feature_maps.png"} <= {x["filename"] for x in figures}
    assert {x["profile"] for x in figures} <= {"reduced", "publication", "smoke"}


def test_active_manuscript_figure_names():
    expected = {
        "fig1_iv-sweeps.png", "fig2_architecture.png", "fig_temporal_arch.png",
        "fig_temporal_results.png", "fig_storerecall.png", "fig_capability.png",
        "fig_nmnist_separability.png", "fig_liability_asset.png",
    }
    manifest = json.loads((ROOT / "experiments" / "figure_manifest.json").read_text())
    assert {row["filename"] for row in manifest["figures"] if row["tier"] == "main"} == expected
    manuscript = ROOT.parent / "manuscript-SNN" / "main.tex"
    if manuscript.exists():
        active = "\n".join(line for line in manuscript.read_text(encoding="utf-8").splitlines()
                           if not line.lstrip().startswith("%"))
        included = set(re.findall(r"includegraphics[^{}]*\{(?:[^{}]*/)?([^/{}]+\.png)\}", active))
        assert included == expected


def test_vendored_reference_hashes():
    manifest = json.loads((ROOT / "experiments" / "figure_manifest.json").read_text())
    refs = ROOT / "experiments" / "assets" / "references"
    for row in manifest["figures"]:
        target = refs / row["filename"]
        assert target.is_file()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == row["reference_hash"]
    assert len({row["reference_hash"] for row in manifest["figures"]}) == 17


def test_static_training_curves_use_the_real_seed_archive():
    manifest = json.loads((ROOT / "experiments" / "figure_manifest.json").read_text())
    by_name = {row["filename"]: row for row in manifest["figures"]}
    archive_path = ROOT / "data" / "fig6_devicefixed_data.json"
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    for name in ("fig3_msnn-training.png", "fig4_mcsnn-training.png"):
        assert by_name[name]["provenance_class"] == "published-sample-archive"
        assert by_name[name]["claim_status"] == "claimable-from-published-seed-archive"
        assert by_name[name]["data_hash"] == archive_hash
        assert by_name[name]["seeds"] == [0, 1, 2]


def test_static_training_archive_schema_is_rectangular_and_finite():
    import numpy as np

    archive = json.loads((ROOT / "data" / "fig6_devicefixed_data.json").read_text())
    config = archive["config"]
    assert config["seeds"] == [0, 1, 2]
    assert config["epochs"] == 12
    assert config["archs"] == ["fc", "conv"]
    for arch in config["archs"]:
        assert set(archive[arch]) == {"ideal", "memristive_pf", "fault"}
        for condition in archive[arch].values():
            assert set(condition) == {"acc_hist", "loss_hist"}
            for values in condition.values():
                array = np.asarray(values, dtype=float)
                assert array.shape == (3, 12)
                assert np.isfinite(array).all()


def test_homeostasis_quantitative_panels_use_sample_arrays():
    manifest = json.loads((ROOT / "experiments" / "figure_manifest.json").read_text())
    by_name = {row["filename"]: row for row in manifest["figures"]}
    for name in ("fig_confusion-faults.png", "homeostatic_feature_maps.png"):
        assert by_name[name]["provenance_class"] == "live-reduced"
        assert by_name[name]["claim_status"] == "reduced-validation"
        archive = ROOT / "data" / "results" / "homeostasis_sample_archive.npz"
        assert by_name[name]["data_hash"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    notebook = nbformat.read(ROOT / "experiments" / "02_homeostasis.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "run_condition" in source
    assert "targets_no_homeo" in source and "predictions_homeo" in source
    assert "conv2_spike_rates" in source and "MCSNN" in source
    assert 'finish_asset("fig_confusion-faults.png"' not in source
    assert 'finish_asset("homeostatic_feature_maps.png"' not in source


def test_architecture_homeostasis_runs_paired_live_models():
    manifest = json.loads((ROOT / "experiments" / "figure_manifest.json").read_text())
    row = next(item for item in manifest["figures"] if item["filename"] == "fig_homeostasis_arch.png")
    fixture = ROOT / "data" / "fixtures" / "homeostasis_reduced_datasets.npz"
    assert row["provenance_class"] == "live-reduced"
    assert row["data_hash"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert row["seeds"] == [0, 1, 2]
    notebook = nbformat.read(ROOT / "experiments" / "02_homeostasis.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert 'finish_asset("fig_homeostasis_arch.png"' not in source
    assert "MSNN" in source and "TemporalMCSNN" in source
    assert "without_homeostasis" in source and "with_homeostasis" in source
    assert "_bootstrap_mean_ci" in source and "benefit_pp" in source
    assert '"dense_train": 512' in source and '"temporal_train": 256' in source
    assert "homeostasis_threshold=5" in source


def test_homeostasis_reduced_dataset_fixture_is_real_and_balanced():
    import numpy as np

    fixture = np.load(ROOT / "data" / "fixtures" / "homeostasis_reduced_datasets.npz",
                      allow_pickle=False)
    metadata = json.loads(str(fixture["metadata_json"].item()))
    assert metadata["selection_seed"] == 20260711
    for task in ("mnist", "fashion"):
        for split, expected in (("train", 1000), ("test", 500)):
            x, y = fixture[f"{task}_{split}_x"], fixture[f"{task}_{split}_y"]
            assert x.shape == (expected, 1, 28, 28) and x.dtype == np.uint8
            assert np.bincount(y, minlength=10).tolist() == [expected // 10] * 10


def test_homeostasis_sample_archive_has_real_per_example_arrays():
    import numpy as np

    archive = np.load(ROOT / "data" / "results" / "homeostasis_sample_archive.npz", allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"].item()))
    assert metadata["dataset"] == "MNIST test split"
    assert metadata["seed"] == 0 and metadata["test_subset"] == 128
    for arm in ("no_homeo", "homeo"):
        targets = archive[f"targets_{arm}"]
        predictions = archive[f"predictions_{arm}"]
        assert targets.shape == predictions.shape == (128,)
        assert np.unique(targets).size == 10
    rates = archive["conv2_spike_rates"]
    assert rates.shape == (metadata["num_steps"], 8)
    assert np.isfinite(rates).all() and np.count_nonzero(rates) > 0


def test_appendix_generators_replace_approximate_blocks():
    def code(name):
        notebook = nbformat.read(ROOT / "experiments" / name, as_version=4)
        return "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    static = code("01_device_and_static.ipynb")
    homeo = code("02_homeostasis.ipynb")
    temporal = code("03_temporal_memory.ipynb")
    assert "memristive crossbar $G_{ij}$" in static
    assert "ref_windows" in static and "forward-Euler" in static
    assert 'DATA = ROOT / "data" / "fig6_devicefixed_data.json"' in static
    assert "generate_static_archive" in static and "precompute_device_params" in static
    assert "from mnn_torch.models import MSNN, MCSNN" in static
    assert 'for arch in config["archs"]' in static
    assert 'for condition in config["conditions"]' in static
    assert '"acc_hist": [row["acc_hist"]' in static
    assert '"loss_hist": [row["loss_hist"]' in static
    assert 'GENERATE_STATIC_ARCHIVE = os.getenv("MNN_GENERATE_STATIC_ARCHIVE", "0") == "1"' in static
    assert 'SHOW_CONTEXT_FIGURES = os.getenv("MNN_SHOW_CONTEXT_FIGURES", "0") == "1"' in static
    assert 'PLOT_ARCHIVE = generate_static_archive(RUN_PROFILE, output)' in static
    assert 'RUN_PROFILE == "publication" and not GENERATE_STATIC_ARCHIVE' in static
    assert 'PLOT_PROVENANCE = "live-reduced"' in static
    assert "plt.close(fig)\n    return None\n\ndef record_skipped" in static
    assert 'finish(fig, "snn_schematic.png"' in static
    assert 'record_skipped("snn_schematic.png"' not in static
    assert "one measured\\n" in homeo and "homeostatic regulariser" in homeo
    assert "membrane state carried across frames" in temporal
    assert "Aggregate-only store-recall ladder" not in temporal
    assert "notebook schematic" not in static + homeo + temporal


def test_device_notebook_has_no_stale_or_duplicate_embedded_outputs():
    notebook = nbformat.read(ROOT / "experiments" / "01_device_and_static.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is None for cell in code_cells)
    assert all(not cell.outputs for cell in code_cells)


def test_static_generator_publication_budget_matches_archive():
    notebook = nbformat.read(ROOT / "experiments" / "01_device_and_static.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for fragment in (
        '"seeds": [0, 1, 2]', '"epochs": 12', '"num_steps": 10',
        '"batch_size": 64', '"train_subset": 6000', '"test_subset": 2000',
    ):
        assert fragment in source

    for fragment in (
        '"epochs": 6', '"num_steps": 10',
        '"train_subset": 512', '"test_subset": 512',
    ):
        assert fragment in source


def test_representation_live_protocols_match_claimed_evaluations():
    notebook = nbformat.read(ROOT / "experiments" / "04_representations.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert '"nmnist_epochs": 40' in source and '"eeg_epochs": 60' in source
    assert "PCA(n_components=min(20" in source
    assert "StratifiedKFold(folds" in source and "cross_val_predict" in source
    assert "umap_model.transform(features[idx_test])" in source
    assert '0:"auditory/left"' in source and '3:"visual/right"' in source
    assert "temporal_mcsnn.conv2_spikes" in source and "mcsnn.conv2_spikes" in source
    assert "MNN_USE_REPRESENTATION_ARCHIVE" in source
    assert "MNN_DOWNLOAD_REPRESENTATION_DATA" in source
    assert "download_source_datasets" in source
    assert "PLACEHOLDER" not in source


def test_notebooks_are_self_contained_and_saving_is_opt_in():
    for path in (ROOT / "experiments").glob("*.ipynb"):
        text = path.read_text(encoding="utf-8")
        assert "mnn_torch.repro_figures" not in text
        if path.name != "REPRODUCE.ipynb":
            source = "\n".join(
                cell.source for cell in nbformat.read(path, as_version=4).cells
                if cell.cell_type == "code"
            )
            assert 'os.getenv("MNN_SAVE_FIGURES", "0")' in source
            assert '{"placeholder", "reference-only", "external-gated"}' in source
            assert 'SPEC[name]["provenance"]' not in source


def test_pseudo_seed_publication_caches_are_removed():
    forbidden = list((ROOT / "data" / "results" / "storerecall").rglob("*.npy"))
    forbidden += list((ROOT / "data" / "results" / "temporal").rglob("*.npy"))
    assert forbidden == []
    assert not (ROOT / "src" / "mnn_torch" / "repro_figures.py").exists()


def test_temporal_quantitative_panels_run_live_with_optional_strict_archive():
    manifest = json.loads((ROOT / "experiments" / "figure_manifest.json").read_text())
    by_name = {row["filename"]: row for row in manifest["figures"]}
    fixture = ROOT / "data" / "fixtures" / "temporal_reduced_datasets.npz"
    for name in ("fig_temporal_results.png", "fig_storerecall.png", "fig_capability.png"):
        assert by_name[name]["provenance_class"] == "live-reduced"
        assert by_name[name]["claim_status"] == "reduced-validation"
        assert by_name[name]["seeds"] == [0, 1, 2]
        assert by_name[name]["data_hash"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
    notebook = nbformat.read(ROOT / "experiments" / "03_temporal_memory.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "load_seed_archive" in source
    assert "runner_code_sha256" in source and "dataset_sha256" in source
    assert "repeated aggregate mean rejected as seed evidence" in source
    assert "TemporalMSNN" in source and "TemporalMCSNN" in source
    assert "DeviceLeaky" in source and "AdaptiveLeaky" in source
    assert "MNN_USE_TEMPORAL_ARCHIVE" in source
    assert "N-MNIST" in source and "SHD" in source and "DVS128 Gesture" in source
    assert 'agg["retention"]' not in source
    assert 'agg["store_recall"]' not in source
    assert 'agg["temporal_capability"]' not in source


def test_temporal_reduced_fixture_contains_genuine_balanced_event_samples():
    fixture = np.load(ROOT / "data" / "fixtures" / "temporal_reduced_datasets.npz",
                      allow_pickle=False)
    metadata = json.loads(str(fixture["metadata_json"].item()))
    assert metadata["selection_seed"] == 20260711 and metadata["time_bins"] == 6
    for key, classes in (("nmnist", 10), ("shd", 20), ("dvs", 11)):
        x, y = fixture[f"{key}_x"], fixture[f"{key}_y"]
        assert len(x) == len(y) == classes * 20 and x.dtype == np.uint8
        assert np.bincount(y, minlength=classes).tolist() == [20] * classes
        assert np.count_nonzero(x) > 0


def test_representation_figures_run_live_from_balanced_genuine_samples():
    path = ROOT / "data" / "fixtures" / "representations_reduced_datasets.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "a6303dd229e0d0e8e3ac334f59473f81f2677952b2b62daa3755c049c7aeefbb"
    )
    fixture = np.load(path, allow_pickle=False)
    metadata = json.loads(str(fixture["metadata"].item()))
    assert metadata["fixture"] == "representations-reduced-v1"
    assert fixture["nmnist_x"].shape == (600, 10, 2, 34, 34)
    assert np.bincount(fixture["nmnist_y"], minlength=10).tolist() == [60] * 10
    assert fixture["eeg_x"].shape == (256, 60, 70)
    assert np.bincount(fixture["eeg_y"], minlength=4).tolist() == [64] * 4
    assert np.count_nonzero(fixture["nmnist_x"]) > 0
    assert np.isfinite(fixture["eeg_x"]).all()
    manifest = json.loads((ROOT / "experiments" / "figure_manifest.json").read_text())
    by_name = {row["filename"]: row for row in manifest["figures"]}
    for name in ("fig_nmnist_separability.png", "fig6_eeg-embeddings.png"):
        assert by_name[name]["provenance_class"] == "live-reduced"
        assert by_name[name]["claim_status"] == "reduced-validation"
        assert by_name[name]["seeds"] == [0]
        assert by_name[name]["data_hash"] == hashlib.sha256(path.read_bytes()).hexdigest()
