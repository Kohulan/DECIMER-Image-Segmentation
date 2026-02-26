"""
Test helper functions for DECIMER Segmentation tests.
"""

import tempfile
import pymupdf


def create_test_pdf(num_pages: int = 1) -> str:
    """
    Create a temporary PDF with black rectangles (simulating chemical structures).

    Args:
        num_pages: Number of pages to create

    Returns:
        Path to temporary PDF file (caller must delete)
    """
    f = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf = pymupdf.open()

    for i in range(num_pages):
        page = pdf.new_page(width=500, height=500)
        # Place structure at different positions on each page
        x_offset = 50 + (i * 100) % 300
        y_offset = 50 + (i * 100) % 300
        page.draw_rect(
            pymupdf.Rect(x_offset, y_offset, x_offset + 100, y_offset + 100),
            color=(0, 0, 0),
            fill=(0, 0, 0),
        )

    pdf.save(f.name)
    pdf.close()
    f.close()
    return f.name
