"""Tests for downloading and loading the model on first use instead of at import.

None of these tests needs the network or the real weights: the Mask R-CNN class and the
download function are replaced by recorders.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import decimer_segmentation
from decimer_segmentation import decimer_segmentation as ds

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    """A model that finds nothing."""

    def __init__(self):
        self.detect_calls = 0

    def detect(self, images, verbose=0):
        self.detect_calls += 1
        height, width = images[0].shape[:2]
        return [
            {
                "masks": np.zeros((height, width, 0), dtype=bool),
                "rois": np.zeros((0, 4), dtype=int),
                "scores": np.zeros((0,), dtype=float),
            }
        ]


@pytest.fixture
def fake_loader(monkeypatch):
    """Replace the code that builds the model, and start and end with no model loaded."""
    record = SimpleNamespace(loads=0, delay=0.0, models=[])

    def load():
        record.loads += 1
        time.sleep(record.delay)
        model = FakeModel()
        record.models.append(model)
        return model

    monkeypatch.setattr(ds, "_load_model_internal", load)
    monkeypatch.setattr(ds, "_warmup_model", lambda model: None)
    ds.reset_model()
    yield record
    ds.reset_model()


def test_import_downloads_nothing_and_loads_no_model(tmp_path):
    """`import decimer_segmentation` in a fresh interpreter: no network, no model, no files."""
    script = """
import os
import pathlib
import socket


def refuse(*args, **kwargs):
    raise AssertionError("network access while importing decimer_segmentation")


socket.socket.connect = refuse
socket.create_connection = refuse
socket.getaddrinfo = refuse

import decimer_segmentation
from decimer_segmentation import decimer_segmentation as ds

assert ds._model is None
assert not ds._model_warmed_up
assert "segmentation_model" not in vars(ds)
assert not hasattr(ds, "no_such_name")
home = pathlib.Path(os.environ["PYSTOW_HOME"])
assert not (home / "DECIMER-Segmentation_model").exists()
assert not list(home.rglob("*.h5"))
print("import-is-passive")
"""
    env = dict(os.environ)
    env["PYSTOW_HOME"] = str(tmp_path / "pystow-home")
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "import-is-passive" in result.stdout


def test_model_is_loaded_by_the_first_segmentation_and_kept(fake_loader):
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    assert fake_loader.loads == 0

    assert ds.segment_chemical_structures(image) == []
    assert fake_loader.loads == 1

    assert ds.segment_chemical_structures(image) == []
    assert fake_loader.loads == 1
    assert fake_loader.models[0].detect_calls == 2


def test_get_model_loads_once(fake_loader):
    first = ds.get_model()
    assert ds.get_model() is first
    assert fake_loader.loads == 1


def test_concurrent_first_calls_load_once(fake_loader):
    fake_loader.delay = 0.3
    count = 4
    barrier = threading.Barrier(count)
    models = []

    def first_call():
        barrier.wait(timeout=30)
        models.append(ds.get_model())

    threads = [threading.Thread(target=first_call) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert fake_loader.loads == 1
    assert len(models) == count
    assert all(model is models[0] for model in models)


def test_segmentation_model_attribute_of_older_releases_still_resolves(fake_loader):
    """`decimer_segmentation.segmentation_model` was set at import up to version 1.5.1."""
    assert fake_loader.loads == 0

    model = ds.segmentation_model
    assert model is fake_loader.models[0]
    assert decimer_segmentation.segmentation_model is model
    assert fake_loader.loads == 1


def test_unknown_attributes_still_raise_attribute_error(fake_loader):
    with pytest.raises(AttributeError):
        ds.no_such_name
    with pytest.raises(AttributeError):
        decimer_segmentation.no_such_name
    assert not hasattr(ds, "no_such_name")
    assert fake_loader.loads == 0


def test_reset_model_makes_the_next_call_load_again(fake_loader):
    """set_inference_config() asks for this: it only applies to a model loaded afterwards."""
    ds.get_model()
    ds.reset_model()
    ds.get_model()
    assert fake_loader.loads == 2


class FakeMaskRCNN:
    created = []

    def __init__(self, mode, model_dir, config):
        self.mode = mode
        self.weights = None
        FakeMaskRCNN.created.append(self)

    def load_weights(self, path, by_name=False):
        self.weights = path


@pytest.fixture
def real_loader(tmp_path, monkeypatch):
    """The real _load_model_internal, with a fake network and a fake Mask R-CNN."""
    home = tmp_path / "pystow-home"
    monkeypatch.setenv("PYSTOW_HOME", str(home))
    weights = home / "DECIMER-Segmentation_model" / ds.MODEL_FILENAME
    record = SimpleNamespace(weights=weights, downloads=[], error=None)

    def fake_download(model_url, model_path, verbose=1):
        record.downloads.append(model_url)
        if record.error is not None:
            raise record.error
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"placeholder")

    FakeMaskRCNN.created = []
    monkeypatch.setattr(ds, "download_trained_weights", fake_download)
    monkeypatch.setattr(ds.modellib, "MaskRCNN", FakeMaskRCNN)
    monkeypatch.setattr(ds, "_warmup_model", lambda model: None)
    ds.reset_model()
    yield record
    ds.reset_model()


def test_first_use_downloads_missing_weights_once(real_loader):
    model = ds.get_model()
    assert real_loader.downloads == [ds.MODEL_DOWNLOAD_URL]
    assert model.weights == str(real_loader.weights)

    assert ds.get_model() is model
    assert real_loader.downloads == [ds.MODEL_DOWNLOAD_URL]
    assert len(FakeMaskRCNN.created) == 1


def test_existing_weights_are_used_without_network(real_loader):
    real_loader.weights.parent.mkdir(parents=True)
    real_loader.weights.write_bytes(b"placeholder")
    real_loader.error = AssertionError("downloaded although the weights exist")

    model = ds.get_model()

    assert real_loader.downloads == []
    assert model.weights == str(real_loader.weights)


def test_pystow_home_set_after_import_is_respected(real_loader, tmp_path, monkeypatch):
    elsewhere = tmp_path / "baked-in-image"
    weights = elsewhere / "DECIMER-Segmentation_model" / ds.MODEL_FILENAME
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"placeholder")
    monkeypatch.setenv("PYSTOW_HOME", str(elsewhere))
    real_loader.error = AssertionError("downloaded although the weights exist")

    assert ds.get_model().weights == str(weights)
    assert real_loader.downloads == []


def test_failed_download_is_not_cached_and_is_retried(real_loader):
    real_loader.error = OSError("zenodo is down")
    with pytest.raises(OSError, match="zenodo is down"):
        ds.get_model()

    real_loader.error = None
    model = ds.get_model()
    assert model.weights == str(real_loader.weights)
    assert real_loader.downloads == [ds.MODEL_DOWNLOAD_URL, ds.MODEL_DOWNLOAD_URL]
