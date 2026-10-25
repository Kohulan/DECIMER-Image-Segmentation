"""Tests for how PDF pages are read: pypdfium2 by default, PyMuPDF (AGPL-3.0) only when asked for.

pypdfium2 (Apache-2.0 / BSD-3-Clause) is a dependency of the package, so the default backend is tested
everywhere, with PyMuPDF made absent. The tests of the "pymupdf" backend that need PyMuPDF are skipped
where it is not installed (install it with the `pdf` extra); the ones that simulate it missing run anywhere.

The PDFs are written by hand (see make_pdf), so that no PDF library is needed to build them.
"""

from __future__ import annotations

import importlib.util
import os
import runpy
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import setuptools
from packaging.requirements import Requirement

from decimer_segmentation import decimer_segmentation as ds

REPO_ROOT = Path(__file__).resolve().parents[1]

needs_pymupdf = pytest.mark.skipif(
    importlib.util.find_spec("pymupdf") is None,
    reason='PyMuPDF is not installed (pip install "decimer-segmentation[pdf]")',
)
BACKENDS = ["pdfium", pytest.param("pymupdf", marks=needs_pymupdf)]


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


@pytest.fixture(autouse=True)
def fresh_pdf_lock(monkeypatch):
    """A test of broken code that leaks the lock must fail, not hang every later test."""
    monkeypatch.setattr(ds, "_pdf_lock", threading.Lock())


@pytest.fixture
def without_pymupdf(monkeypatch):
    """Make `import pymupdf` fail, as it does where PyMuPDF is not installed."""
    monkeypatch.setitem(sys.modules, "pymupdf", None)
    monkeypatch.setitem(sys.modules, "fitz", None)


def make_pdf(path: Path, pages: int, size=(595, 842)) -> None:
    """A white PDF whose page i has a red 50 x 50 pt square 100 pt from the top and 100 + 100 * i from the left."""
    width, height = size
    kids = " ".join(f"{3 + 2 * index} 0 R" for index in range(pages))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode(),
    ]
    for index in range(pages):
        content = f"1 0 0 rg {100 + 100 * index} {height - 150} 50 50 re f".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Contents {4 + 2 * index} 0 R >>".encode()
        )
        objects.append(
            b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream"
        )

    document = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(document))
        document += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref_offset = len(document)
    document += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        document += b"%010d 00000 n \n" % offset
    document += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_offset,
    )
    path.write_bytes(bytes(document))


def assert_pages_in_order_in_bgr(images, pages):
    """The red square of every page is where it is expected, in that page only: in order, BGR."""
    assert len(images) == pages
    row = int(125 * 300 / 72)
    for index, image in enumerate(images):
        assert image.dtype == np.uint8
        assert image.ndim == 3 and image.shape[2] == 3
        assert image.shape[0] > 3000  # A4 at 300 DPI is about 2480 x 3508
        for other in range(pages):
            column = int((125 + 100 * other) * 300 / 72)
            expected = [0, 0, 255] if other == index else [255, 255, 255]
            assert image[row, column].tolist() == expected


@pytest.mark.parametrize("pymupdf", ["as-installed", "absent"])
def test_the_package_never_imports_pymupdf_unless_asked_to(tmp_path, pymupdf):
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

# Reading a PDF with the default backend does not look for PyMuPDF either ...
assert len(ds._load_pdf_pages(sys.argv[2])) == 2
assert not attempts, attempts
assert "pymupdf" not in sys.modules

# ... and asking for PyMuPDF is what makes the import happen (so the watcher does see imports).
try:
    ds._load_pdf_pages(sys.argv[2], "pymupdf")
except ImportError:
    pass  # PyMuPDF is made absent, or is not installed
else:
    assert not block
assert attempts and attempts[0] == "pymupdf", attempts
print("import-ok")
"""
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf, pages=2)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYSTOW_HOME"] = str(tmp_path / "pystow-home")
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    result = subprocess.run(
        [sys.executable, "-c", script, pymupdf, str(pdf)],
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

    # Whatever backend is named: it only matters for PDFs.
    for backend in ds.PDF_BACKENDS:
        fake_model.detect_calls = 0
        assert ds.segment_chemical_structures_from_file(
            str(path), expand=False, return_bboxes=True, pdf_backend=backend
        ) == ([], [])
        assert fake_model.detect_calls == 1


def test_numpy_arrays_work_without_pymupdf(without_pymupdf, fake_model):
    image = np.full((300, 300, 3), 255, dtype=np.uint8)
    assert ds.segment_chemical_structures(image, expand=False) == []
    assert fake_model.detect_calls == 1


@pytest.mark.parametrize("name", ["paper.pdf", "PAPER.PDF"])
@pytest.mark.parametrize("pages", [1, 3])
def test_pdf_files_work_by_default_without_pymupdf(
    without_pymupdf, fake_model, tmp_path, name, pages
):
    path = tmp_path / name
    make_pdf(path, pages)

    images = ds._load_pdf_pages(str(path))
    assert_pages_in_order_in_bgr(images, pages)

    assert ds.segment_chemical_structures_from_file(
        str(path), expand=False, return_bboxes=True
    ) == ([], [])
    assert fake_model.detect_calls == pages


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", ["paper.pdf", "PAPER.PDF"])
@pytest.mark.parametrize("pages", [1, 3])
def test_pages_come_in_order_in_bgr_with_either_backend(
    fake_model, tmp_path, backend, name, pages
):
    path = tmp_path / name
    make_pdf(path, pages)

    assert_pages_in_order_in_bgr(ds._load_pdf_pages(str(path), backend), pages)

    assert ds.segment_chemical_structures_from_file(
        str(path), expand=False, return_bboxes=True, pdf_backend=backend
    ) == ([], [])
    assert fake_model.detect_calls == pages


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "size, shape",
    [
        ((595, 842), (3509, 2480, 3)),
        ((612, 792), (3300, 2550, 3)),  # US Letter: 11 in is exactly 3300 px at 300 DPI
        ((792, 1224), (5100, 3300, 3)),  # Tabloid
    ],
)
def test_pages_are_exactly_300_dpi(tmp_path, backend, size, shape):
    """pdfium rounds a size of 3300.0000000000005 px up to 3301: both backends must agree."""
    path = tmp_path / "paper.pdf"
    make_pdf(path, 1, size)

    (image,) = ds._load_pdf_pages(str(path), backend)

    assert image.shape == shape


@pytest.mark.parametrize("backend", BACKENDS)
def test_pages_own_their_pixels(tmp_path, backend):
    """Not a view of a bitmap that goes away with the document."""
    path = tmp_path / "paper.pdf"
    make_pdf(path, 2)

    images = ds._load_pdf_pages(str(path), backend)

    for image in images:
        assert image.flags.owndata and image.flags.c_contiguous
        assert image.flags.writeable


@pytest.fixture
def spy_on_backend(monkeypatch):
    """Record how a backend is used: documents opened, pages rendered (thread, index), documents closed."""

    def install(backend):
        record = SimpleNamespace(opened=0, rendered=[], closed=0)
        real_open = getattr(ds, f"_open_pdf_{backend}")
        real_render = getattr(ds, f"_render_page_{backend}")

        class Document:
            def __init__(self, document):
                self._document = document

            def __len__(self):
                return len(self._document)

            def __getitem__(self, index):
                return self._document[index]

            def close(self):
                record.closed += 1
                self._document.close()

        def open_pdf(pdf_path):
            record.opened += 1
            return Document(real_open(pdf_path))

        def render_page(document, index):
            record.rendered.append((threading.get_ident(), index))
            return real_render(document, index)

        monkeypatch.setattr(ds, f"_open_pdf_{backend}", open_pdf)
        monkeypatch.setattr(ds, f"_render_page_{backend}", render_page)
        return record

    return install


@pytest.mark.parametrize("backend", BACKENDS)
def test_pages_are_rendered_one_after_the_other_from_one_document_that_is_closed(
    spy_on_backend, tmp_path, backend
):
    """Neither PDFium nor PyMuPDF is thread-safe: no thread pool, one open document, closed afterwards."""
    path = tmp_path / "paper.pdf"
    make_pdf(path, 3)
    record = spy_on_backend(backend)

    ds._load_pdf_pages(str(path), backend)

    assert record.opened == 1
    assert record.rendered == [(threading.get_ident(), index) for index in range(3)]
    assert record.closed == 1
    assert not ds._pdf_lock.locked()


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_document_is_closed_and_the_lock_released_when_a_page_fails(
    monkeypatch, spy_on_backend, tmp_path, backend
):
    path = tmp_path / "paper.pdf"
    make_pdf(path, 3)
    record = spy_on_backend(backend)
    render_page = getattr(ds, f"_render_page_{backend}")

    def failing_on_page_two(document, index):
        if index == 1:
            raise RuntimeError("page 2 is damaged")
        return render_page(document, index)

    monkeypatch.setattr(ds, f"_render_page_{backend}", failing_on_page_two)

    with pytest.raises(RuntimeError, match="page 2 is damaged"):
        ds._load_pdf_pages(str(path), backend)

    assert record.opened == 1
    assert record.closed == 1
    assert not ds._pdf_lock.locked()


def test_concurrent_callers_render_one_pdf_at_a_time(monkeypatch):
    """The pdf libraries are not thread-safe, so two threads must never be inside them together."""
    state = SimpleNamespace(active=0, peak=0, guard=threading.Lock())

    class FakeDocument:
        def __len__(self):
            return 3

        def close(self):
            pass

    def render_page(document, index):
        with state.guard:
            state.active += 1
            state.peak = max(state.peak, state.active)
        time.sleep(0.02)
        with state.guard:
            state.active -= 1
        return np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(ds, "_open_pdf_pdfium", lambda path: FakeDocument())
    monkeypatch.setattr(ds, "_render_page_pdfium", render_page)

    threads = [
        threading.Thread(target=ds._load_pdf_pages, args=("paper.pdf",))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert state.peak == 1
    assert not any(thread.is_alive() for thread in threads)


@pytest.mark.parametrize("name", ["paper.pdf", "PAPER.PDF"])
def test_pymupdf_backend_without_pymupdf_raises_an_import_error_that_helps(
    without_pymupdf, fake_model, tmp_path, name
):
    path = tmp_path / name
    make_pdf(path, 1)

    with pytest.raises(ImportError) as raised:
        ds.segment_chemical_structures_from_file(str(path), pdf_backend="pymupdf")

    message = str(raised.value)
    assert 'pdf_backend="pymupdf"' in message
    assert 'pip install "decimer-segmentation[pdf]"' in message
    assert "PyMuPDF" in message
    assert "AGPL" in message
    assert "pypdfium2" in message  # PDFs do not need PyMuPDF: say what is used instead
    assert isinstance(raised.value.__cause__, ImportError)
    assert fake_model.detect_calls == 0


def test_the_pymupdf_error_comes_before_any_model_is_loaded(
    without_pymupdf, monkeypatch, tmp_path
):
    """Nobody should wait for a 272 MB download to be told that PyMuPDF is missing."""

    def refuse():
        raise AssertionError("the model was loaded before PDF support was checked")

    monkeypatch.setattr(ds, "get_model", refuse)
    path = tmp_path / "paper.pdf"
    make_pdf(path, 1)

    with pytest.raises(ImportError):
        ds.segment_chemical_structures_from_file(str(path), pdf_backend="pymupdf")


@pytest.mark.parametrize("backend", ["poppler", "PyMuPDF", "fitz", "", None])
@pytest.mark.parametrize("name", ["paper.pdf", "page.png"])
def test_an_unknown_backend_is_a_value_error_before_the_model_is_touched(
    monkeypatch, tmp_path, backend, name
):
    def refuse():
        raise AssertionError("the model was loaded for an unknown backend")

    monkeypatch.setattr(ds, "get_model", refuse)
    path = tmp_path / name
    if name.endswith(".pdf"):
        make_pdf(path, 1)
    else:
        assert cv2.imwrite(str(path), np.full((50, 50, 3), 255, dtype=np.uint8))

    with pytest.raises(ValueError) as raised:
        ds.segment_chemical_structures_from_file(str(path), pdf_backend=backend)

    assert "'pdfium'" in str(raised.value) and "'pymupdf'" in str(raised.value)
    with pytest.raises(ValueError):
        ds._load_pdf_pages(str(path), backend)


def test_pdfium_is_the_default_backend():
    assert ds.DEFAULT_PDF_BACKEND == "pdfium"
    assert ds.PDF_BACKENDS == ("pdfium", "pymupdf")


def test_the_backends_are_part_of_the_package_namespace():
    import decimer_segmentation

    assert decimer_segmentation.PDF_BACKENDS == ds.PDF_BACKENDS
    assert decimer_segmentation.DEFAULT_PDF_BACKEND == ds.DEFAULT_PDF_BACKEND


@pytest.fixture
def recorded_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(ds, "get_model", lambda: None)

    def segment(path, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(ds, "segment_chemical_structures_from_file", segment)
    return calls


@pytest.mark.parametrize(
    "arguments, expected",
    [([], "pdfium"), (["--pdf-backend", "pymupdf"], "pymupdf")],
)
def test_the_command_line_chooses_the_backend(
    monkeypatch, recorded_calls, arguments, expected
):
    monkeypatch.setattr(
        sys, "argv", ["decimer-segmentation", "-i", "x.pdf", *arguments]
    )

    ds.main()

    assert [call["pdf_backend"] for call in recorded_calls] == [expected]


def test_the_command_line_rejects_an_unknown_backend(monkeypatch, recorded_calls):
    monkeypatch.setattr(
        sys, "argv", ["decimer-segmentation", "-i", "x.pdf", "--pdf-backend", "poppler"]
    )

    with pytest.raises(SystemExit) as raised:
        ds.main()

    assert raised.value.code == 2
    assert recorded_calls == []


@pytest.mark.parametrize(
    "arguments, expected",
    [([], "pdfium"), (["--pdf_backend", "pymupdf"], "pymupdf")],
)
def test_the_document_script_chooses_the_backend(
    monkeypatch, tmp_path, arguments, expected
):
    calls = []

    def segment(path, **kwargs):
        calls.append(kwargs)
        return []

    script = runpy.run_path(
        str(REPO_ROOT / "segment_structures_in_document.py"), run_name="not_main"
    )
    main = script["main"]
    main.__globals__["segment_chemical_structures_from_file"] = segment
    path = tmp_path / "paper.pdf"
    make_pdf(path, 1)
    monkeypatch.setattr(sys, "argv", ["script", str(path), *arguments])

    main()

    assert [call["pdf_backend"] for call in calls] == [expected]


def test_setup_py_needs_pypdfium2_and_makes_pymupdf_an_optional_extra(monkeypatch):
    captured = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    monkeypatch.chdir(REPO_ROOT)

    runpy.run_path(str(REPO_ROOT / "setup.py"))

    required = {
        Requirement(text).name.lower(): Requirement(text)
        for text in captured["install_requires"]
    }
    extra = {
        Requirement(text).name.lower() for text in captured["extras_require"]["pdf"]
    }
    assert "pymupdf" not in required
    assert "pymupdf" in extra
    # The default reader is always installed, with a lower bound that has been tested
    # (older 4.x releases work too, but 4.30.0 is the last 4.x feature release).
    assert "pypdfium2" in required
    assert "pypdfium2" not in extra
    assert "4.30.0" in required["pypdfium2"].specifier
    assert "3.0.0" not in required["pypdfium2"].specifier
