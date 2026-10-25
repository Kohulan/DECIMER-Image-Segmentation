"""Tests for the page numbers of segment_chemical_structures_from_file (return_page_numbers).

The structures are red squares in a PDF, and a different number of them is on every page (see
make_structure_pdf in tests/helpers.py). The model is replaced by a detector that finds the red squares in
the page image it is given. Where a square is tells which page it is on, so the expected page numbers and
boxes follow from the layout of the PDF, not from the code that is tested: a segment that is paired with
the wrong page, pages in the wrong order, or a page number counted from 0 cannot pass.

The PDFs are written by hand, so these tests need no PDF library; those of the "pymupdf" backend are
skipped where PyMuPDF is not installed.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from decimer_segmentation import decimer_segmentation as ds
from tests.helpers import BACKENDS, make_structure_pdf

REPO_ROOT = Path(__file__).resolve().parents[1]

PIXELS_PER_POINT = 300 / 72
SQUARE = 50 * PIXELS_PER_POINT
# The two PDF backends draw the edge of a square a pixel apart (anti-aliasing).
TOLERANCE = 2

# How many structures are on each page. Pages without structures are in the middle and at the start, so that
# a page number that counts only the pages with structures, or that starts at the first structure, fails.
LAYOUTS = [[3], [1, 1, 1], [2, 0, 3, 1], [0, 2]]

COMBINATIONS = [
    pytest.param(False, False, id="segments"),
    pytest.param(True, False, id="bboxes"),
    pytest.param(False, True, id="page-numbers"),
    pytest.param(True, True, id="bboxes-and-page-numbers"),
]


class RedSquareModel:
    """Finds the red squares in a page, like the Mask R-CNN finds the structures."""

    def detect(self, images, verbose=0):
        image = images[0]
        blue, green, red = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        is_red = ((red > 200) & (green < 100) & (blue < 100)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(is_red)
        labels_found = np.arange(1, count)
        return [
            {
                "masks": labels[:, :, None] == labels_found,
                "rois": np.array(
                    [
                        [y, x, y + height, x + width]
                        for x, y, width, height, _ in stats[1:]
                    ],
                    dtype=int,
                ).reshape(-1, 4),
                "scores": np.ones(len(labels_found)),
            }
        ]


@pytest.fixture
def red_square_model(monkeypatch):
    monkeypatch.setattr(ds, "get_model", RedSquareModel)


def expected_structures(layout):
    """(page number, top, left in pixels) of every structure, in reading order: page by page, top to bottom."""
    return [
        (
            page,
            (100 + 100 * row) * PIXELS_PER_POINT,
            (100 + 100 * (page - 1)) * PIXELS_PER_POINT,
        )
        for page, structures in enumerate(layout, start=1)
        for row in range(structures)
    ]


def assert_found_as_laid_out(segments, bboxes, page_numbers, layout):
    """Every segment is the square that was put on its page, and is paired with that page's number."""
    expected = expected_structures(layout)
    assert page_numbers == [page for page, _, _ in expected]
    assert all(type(page) is int for page in page_numbers)
    assert len(segments) == len(bboxes) == len(expected)
    for segment, bbox, (page, top, left) in zip(segments, bboxes, expected):
        y0, x0, y1, x1 = bbox
        assert abs(y0 - top) <= TOLERANCE, f"page {page}: {bbox}"
        assert abs(x0 - left) <= TOLERANCE, f"page {page}: {bbox}"
        assert abs((y1 - y0) - SQUARE) <= TOLERANCE, f"page {page}: {bbox}"
        assert abs((x1 - x0) - SQUARE) <= TOLERANCE, f"page {page}: {bbox}"
        assert segment.shape == (y1 - y0, x1 - x0, 4)
        centre = segment[segment.shape[0] // 2, segment.shape[1] // 2]
        assert centre.tolist() == [0, 0, 255, 255]  # red, in BGRA


def unpack(result, return_bboxes, return_page_numbers):
    """Split a return value into (segments, bboxes, page_numbers): None for what was not asked for."""
    wanted = [True, return_bboxes, return_page_numbers]
    if sum(wanted) == 1:
        assert type(result) is list
        parts = [result]
    else:
        assert type(result) is tuple and len(result) == sum(wanted)
        parts = list(result)
    parts = iter(parts)
    return tuple(next(parts) if flag else None for flag in wanted)


def write_image_with_squares(path, squares=2):
    image = np.full((600, 800, 3), 255, dtype=np.uint8)
    for index in range(squares):
        left = 50 + 250 * index
        cv2.rectangle(image, (left, 100), (left + 99, 199), (0, 0, 255), -1)
    assert cv2.imwrite(str(path), image)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("layout", LAYOUTS, ids=str)
def test_every_structure_is_paired_with_the_page_it_is_on(
    red_square_model, tmp_path, backend, layout
):
    path = tmp_path / "paper.pdf"
    make_structure_pdf(path, layout)

    segments, bboxes, page_numbers = ds.segment_chemical_structures_from_file(
        str(path),
        expand=False,
        return_bboxes=True,
        return_page_numbers=True,
        pdf_backend=backend,
    )

    assert_found_as_laid_out(segments, bboxes, page_numbers, layout)


def test_page_numbers_need_no_pymupdf(without_pymupdf, red_square_model, tmp_path):
    layout = [2, 0, 3, 1]
    path = tmp_path / "paper.pdf"
    make_structure_pdf(path, layout)

    segments, bboxes, page_numbers = ds.segment_chemical_structures_from_file(
        str(path), expand=False, return_bboxes=True, return_page_numbers=True
    )

    assert_found_as_laid_out(segments, bboxes, page_numbers, layout)


@pytest.mark.parametrize("return_bboxes, return_page_numbers", COMBINATIONS)
def test_the_flags_only_choose_what_is_returned(
    red_square_model, tmp_path, return_bboxes, return_page_numbers
):
    layout = [2, 0, 1]
    path = tmp_path / "paper.pdf"
    make_structure_pdf(path, layout)
    reference = ds.segment_chemical_structures_from_file(
        str(path), expand=False, return_bboxes=True, return_page_numbers=True
    )

    result = ds.segment_chemical_structures_from_file(
        str(path),
        expand=False,
        return_bboxes=return_bboxes,
        return_page_numbers=return_page_numbers,
    )

    segments, bboxes, page_numbers = unpack(result, return_bboxes, return_page_numbers)
    assert len(segments) == 3
    for segment, expected in zip(segments, reference[0]):
        assert np.array_equal(segment, expected)
    if return_bboxes:
        assert bboxes == reference[1]
    if return_page_numbers:
        assert page_numbers == [1, 1, 3]


@pytest.mark.parametrize("name", ["page.png", "PAGE.PNG", "page.bmp"])
@pytest.mark.parametrize("return_bboxes, return_page_numbers", COMBINATIONS)
def test_every_structure_in_an_image_file_is_on_page_1(
    red_square_model, tmp_path, name, return_bboxes, return_page_numbers
):
    path = tmp_path / name
    write_image_with_squares(path, squares=2)

    result = ds.segment_chemical_structures_from_file(
        str(path),
        expand=False,
        return_bboxes=return_bboxes,
        return_page_numbers=return_page_numbers,
    )

    segments, bboxes, page_numbers = unpack(result, return_bboxes, return_page_numbers)
    assert len(segments) == 2
    if return_bboxes:
        assert [(y0, x0) for y0, x0, _, _ in bboxes] == [(100, 50), (100, 300)]
    if return_page_numbers:
        assert page_numbers == [1, 1]


@pytest.mark.parametrize(
    "return_bboxes, return_page_numbers, expected",
    [
        (False, False, []),
        (True, False, ([], [])),
        (False, True, ([], [])),
        (True, True, ([], [], [])),
    ],
    ids=["segments", "bboxes", "page-numbers", "bboxes-and-page-numbers"],
)
@pytest.mark.parametrize("source", ["pdf", "image"])
def test_a_file_without_structures_gives_empty_lists_in_the_same_shape(
    red_square_model, tmp_path, source, return_bboxes, return_page_numbers, expected
):
    if source == "pdf":
        path = tmp_path / "paper.pdf"
        make_structure_pdf(path, [0, 0])
    else:
        path = tmp_path / "page.png"
        write_image_with_squares(path, squares=0)

    result = ds.segment_chemical_structures_from_file(
        str(path),
        expand=False,
        return_bboxes=return_bboxes,
        return_page_numbers=return_page_numbers,
    )

    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize(
    "return_bboxes, return_page_numbers, expected",
    [
        (False, False, []),
        (True, False, ([], [])),
        (False, True, ([], [])),
        (True, True, ([], [], [])),
    ],
    ids=["segments", "bboxes", "page-numbers", "bboxes-and-page-numbers"],
)
def test_a_file_without_pages_gives_empty_lists_and_a_warning(
    monkeypatch, caplog, tmp_path, return_bboxes, return_page_numbers, expected
):
    """pdfium and PyMuPDF refuse a PDF without pages, so there is no such file to read: the loader is replaced."""
    path = tmp_path / "paper.pdf"
    make_structure_pdf(path, [1])
    monkeypatch.setattr(ds, "_load_images_from_file", lambda *args, **kwargs: [])
    monkeypatch.setattr(ds, "get_model", lambda: pytest.fail("no page to look at"))

    with caplog.at_level(logging.WARNING, logger=ds.logger.name):
        result = ds.segment_chemical_structures_from_file(
            str(path),
            expand=False,
            return_bboxes=return_bboxes,
            return_page_numbers=return_page_numbers,
        )

    assert result == expected
    assert type(result) is type(expected)
    assert "No images could be extracted" in caplog.text


def test_the_test_helpers_need_no_pymupdf(tmp_path):
    """Importing tests/helpers.py must work where PyMuPDF is not installed (it is an optional dependency)."""
    script = (
        "import sys; sys.modules['pymupdf'] = None; sys.modules['fitz'] = None; "
        "sys.path.insert(0, sys.argv[1]); import tests.helpers"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(REPO_ROOT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stderr[-3000:]
