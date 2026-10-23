"""Tests for PyMuPDF (AGPL-3.0) being an optional dependency that is only used to read PDFs.

The tests that simulate a missing PyMuPDF run in any environment. The test of the PDF path
itself needs PyMuPDF and is skipped without it (install it with the `pdf` extra).
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import setuptools

from decimer_segmentation import decimer_segmentation as ds

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    """A model that finds nothing, and counts how often it is asked."""

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
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(ds, "get_model", lambda: model)
    return model


@pytest.fixture
def without_pymupdf(monkeypatch):
    """Make `import pymupdf` fail, as it does where PyMuPDF is not installed."""
    monkeypatch.setitem(sys.modules, "pymupdf", None)
    monkeypatch.setitem(sys.modules, "fitz", None)


@pytest.mark.parametrize("pymupdf", ["as-installed", "absent"])
def test_importing_the_package_never_imports_pymupdf(tmp_path, pymupdf):
    """Run in a fresh interpreter, with PyMuPDF as installed or made absent, and watch for imports."""
    script = """
import importlib.abc
import sys

attempts = []
block = sys.argv[1] == "absent"


class Watcher(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("pymupdf", "fitz"):
            attempts.append(name)
            if block:
                raise ModuleNotFoundError("No module named " + repr(name), name=name)
        return None


sys.meta_path.insert(0, Watcher())

import decimer_segmentation
from decimer_segmentation import decimer_segmentation as ds

assert not attempts, attempts
assert "pymupdf" not in sys.modules
assert callable(ds.segment_chemical_structures_from_file)
print("import-ok")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYSTOW_HOME"] = str(tmp_path / "pystow-home")
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    result = subprocess.run(
        [sys.executable, "-c", script, pymupdf],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "import-ok" in result.stdout


def test_image_files_work_without_pymupdf(without_pymupdf, fake_model, tmp_path):
    image = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 0), 2)
    path = tmp_path / "page.png"
    assert cv2.imwrite(str(path), image)

    assert ds.segment_chemical_structures_from_file(
        str(path), expand=False, return_bboxes=True
    ) == ([], [])
    assert fake_model.detect_calls == 1


def test_numpy_arrays_work_without_pymupdf(without_pymupdf, fake_model):
    image = np.full((300, 300, 3), 255, dtype=np.uint8)
    assert ds.segment_chemical_structures(image, expand=False) == []
    assert fake_model.detect_calls == 1


@pytest.mark.parametrize("name", ["paper.pdf", "PAPER.PDF"])
def test_pdf_files_without_pymupdf_raise_an_import_error_that_names_the_extra(
    without_pymupdf, fake_model, tmp_path, name
):
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(ImportError) as raised:
        ds.segment_chemical_structures_from_file(str(path))

    message = str(raised.value)
    assert 'pip install "decimer-segmentation[pdf]"' in message
    assert "PyMuPDF" in message
    assert isinstance(raised.value.__cause__, ImportError)
    assert fake_model.detect_calls == 0


def test_the_pdf_error_comes_before_any_model_is_loaded(
    without_pymupdf, monkeypatch, tmp_path
):
    """Nobody should wait for a 272 MB download to be told that PyMuPDF is missing."""

    def refuse():
        raise AssertionError("the model was loaded before PDF support was checked")

    monkeypatch.setattr(ds, "get_model", refuse)
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(ImportError):
        ds.segment_chemical_structures_from_file(str(path))


def make_pdf(path: Path, pages: int) -> None:
    """A PDF whose page i has a red square at x = 100 + 100 * i."""
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page()
        left = 100 + 100 * index
        page.draw_rect(
            pymupdf.Rect(left, 100, left + 50, 150), color=None, fill=(1, 0, 0)
        )
    document.save(str(path))
    document.close()


@pytest.mark.parametrize("pages", [1, 3])
def test_pdf_files_work_with_pymupdf(fake_model, tmp_path, pages):
    path = tmp_path / "paper.pdf"
    make_pdf(path, pages)

    images = ds._load_pdf_pages(str(path))

    assert len(images) == pages
    for index, image in enumerate(images):
        assert image.dtype == np.uint8
        assert image.ndim == 3 and image.shape[2] == 3
        assert image.shape[0] > 3000  # A4 at 300 DPI is about 2480 x 3508
        # The red square (BGR: 0, 0, 255) of this page is where it is expected, in this
        # page only: the pages are in order, and the channels are BGR.
        row = int(125 * 300 / 72)
        for other in range(pages):
            column = int((125 + 100 * other) * 300 / 72)
            expected = [0, 0, 255] if other == index else [255, 255, 255]
            assert image[row, column].tolist() == expected

    assert ds.segment_chemical_structures_from_file(
        str(path), expand=False, return_bboxes=True
    ) == ([], [])
    assert fake_model.detect_calls == pages


def test_setup_py_makes_pymupdf_an_optional_extra(monkeypatch):
    captured = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    monkeypatch.chdir(REPO_ROOT)

    runpy.run_path(str(REPO_ROOT / "setup.py"))

    def names(requirements):
        return {
            requirement.split(">")[0].split("=")[0].split("<")[0].strip().lower()
            for requirement in requirements
        }

    assert "pymupdf" not in names(captured["install_requires"])
    assert "pymupdf" in names(captured["extras_require"]["pdf"])
