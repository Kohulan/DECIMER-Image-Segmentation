"""
Test helpers for DECIMER Segmentation tests.

The test PDFs are written by hand, as raw PDF bytes, so that building them needs no PDF
library: PyMuPDF is an optional dependency and must not be needed to run the tests.
"""

import importlib.util
import tempfile
from pathlib import Path
from typing import Sequence, Tuple

import pytest

needs_pymupdf = pytest.mark.skipif(
    importlib.util.find_spec("pymupdf") is None,
    reason='PyMuPDF is not installed (pip install "decimer-segmentation[pdf]")',
)
BACKENDS = ["pdfium", pytest.param("pymupdf", marks=needs_pymupdf)]

A4 = (595, 842)


def write_pdf(
    path: Path, contents: Sequence[bytes], size: Tuple[int, int] = A4
) -> None:
    """Write a PDF with one page per content stream, every page size[0] x size[1] points."""
    width, height = size
    pages = len(contents)
    kids = " ".join(f"{3 + 2 * index} 0 R" for index in range(pages))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode(),
    ]
    for index, content in enumerate(contents):
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
    Path(path).write_bytes(bytes(document))


def make_structure_pdf(
    path: Path, structures_per_page: Sequence[int], size: Tuple[int, int] = A4
) -> None:
    """A white PDF with red 50 x 50 pt squares standing in for chemical structures.

    Page i (counting from 0) has structures_per_page[i] squares, in a column 100 + 100 * i pt
    from the left: the first one 100 pt from the top of the page, each next one 100 pt lower.
    Where a square is tells which page it is on, so a test can check that a structure that was
    found is paired with the right page number.
    """
    height = size[1]
    contents = []
    for index, structures in enumerate(structures_per_page):
        squares = [
            f"1 0 0 rg {100 + 100 * index} {height - 150 - 100 * row} 50 50 re f"
            for row in range(structures)
        ]
        contents.append("\n".join(squares).encode())
    write_pdf(path, contents, size)


def make_pdf(path: Path, pages: int, size: Tuple[int, int] = A4) -> None:
    """A white PDF whose page i has a red 50 x 50 pt square 100 pt from the top and 100 + 100 * i from the left."""
    make_structure_pdf(path, [1] * pages, size)


def create_test_pdf(num_pages: int = 1) -> str:
    """
    Create a temporary PDF with num_pages pages (each with a red square, see make_pdf).

    Returns:
        Path to temporary PDF file (caller must delete)
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name
    make_pdf(Path(pdf_path), num_pages)
    return pdf_path
