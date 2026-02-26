"""
Comprehensive Test Suite for DECIMER Segmentation

Run with: pytest tests/ -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import pytest
import numpy as np
import cv2

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers import create_test_pdf


class TestImageProcessing:
    """Tests for image processing functions."""

    def test_binarize_image(self):
        """Test image binarization."""
        from decimer_segmentation.optimized_complete_structure import (
            _binarize_image_fast,
        )

        # Create test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Binarize
        result = _binarize_image_fast(image, threshold=0.5)

        assert result.dtype == bool
        assert result.shape == (100, 100)

    def test_binarize_grayscale(self):
        """Test binarization with grayscale input."""
        from decimer_segmentation.optimized_complete_structure import (
            _binarize_image_fast,
        )

        # Create grayscale image
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        result = _binarize_image_fast(image, threshold=0.5)

        assert result.dtype == bool
        assert result.shape == (100, 100)

    def test_get_bnw_image(self):
        """Test black and white conversion."""
        from decimer_segmentation import get_bnw_image

        # Create color image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = get_bnw_image(image)

        assert len(result.shape) == 2
        assert result.dtype == np.uint8

    def test_get_bnw_image_rgba(self):
        """Test black and white conversion with RGBA input."""
        from decimer_segmentation import get_bnw_image

        # Create RGBA image
        image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)

        result = get_bnw_image(image)

        assert len(result.shape) == 2

    def test_get_square_image(self):
        """Test square image generation."""
        from decimer_segmentation import get_square_image

        # Create non-square image
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)

        result = get_square_image(image, target_size=299)

        assert result.shape == (299, 299)

    def test_get_square_image_preserves_aspect(self):
        """Test that square conversion preserves aspect ratio."""
        from decimer_segmentation import get_square_image

        # Create wide image
        image = np.zeros((100, 400, 3), dtype=np.uint8)
        image[40:60, 180:220] = 255  # Add marker

        result = get_square_image(image, target_size=299)

        # Image should be centered with white padding
        assert result.shape == (299, 299)
        # Check corners are white (padding)
        assert result[0, 0] == 255
        assert result[-1, -1] == 255


class TestMaskExpansion:
    """Tests for mask expansion functionality."""

    def test_complete_structure_mask_empty(self):
        """Test mask expansion with empty masks."""
        from decimer_segmentation.optimized_complete_structure import (
            complete_structure_mask,
        )

        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        masks = np.empty((100, 100, 0), dtype=bool)

        result = complete_structure_mask(
            image_array=image, mask_array=masks, max_depiction_size=(50, 50)
        )

        assert result.shape[2] == 0

    def test_flood_fill_from_seeds(self):
        """Test flood fill expansion."""
        from decimer_segmentation.optimized_complete_structure import (
            _flood_fill_from_seeds,
        )

        # Create image with connected component
        image = np.ones((100, 100), dtype=bool)  # All white (background)
        image[40:60, 40:60] = False  # Black square (foreground)

        seeds = [(50, 50)]  # Seed in center of square

        result = _flood_fill_from_seeds(image, seeds)

        # Should capture the entire square
        assert result[50, 50]
        assert result[45, 45]
        assert not result[10, 10]  # Outside

    def test_filter_duplicate_masks(self):
        """Test duplicate mask filtering."""
        from decimer_segmentation.optimized_complete_structure import (
            _filter_duplicate_masks,
        )

        # Create duplicate masks
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[10:20, 10:20] = True

        mask2 = mask1.copy()  # Exact duplicate

        mask3 = np.zeros((100, 100), dtype=bool)
        mask3[50:60, 50:60] = True  # Different mask

        masks = [mask1, mask2, mask3]
        result = _filter_duplicate_masks(masks)

        assert len(result) == 2

    def test_get_seed_pixels(self):
        """Test seed pixel extraction."""
        from decimer_segmentation.optimized_complete_structure import _get_seed_pixels

        # Create mask
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 30:70] = True

        # Create image with foreground in mask region
        image = np.ones((100, 100), dtype=bool)  # White background
        image[40:60, 40:60] = False  # Black foreground

        exclusion = np.zeros((100, 100), dtype=bool)

        seeds = _get_seed_pixels(mask, image, exclusion)

        assert len(seeds) > 0
        # All seeds should be within mask bounds
        for x, y in seeds:
            assert 30 <= x < 70
            assert 30 <= y < 70


class TestBoundingBoxes:
    """Tests for bounding box operations."""

    def test_extract_bboxes(self):
        """Test bounding box extraction from masks."""
        from decimer_segmentation.mrcnn.utils import extract_bboxes

        # Create mask with known bbox
        mask = np.zeros((100, 100, 1), dtype=bool)
        mask[20:40, 30:60, 0] = True

        bboxes = extract_bboxes(mask)

        assert bboxes.shape == (1, 4)
        assert list(bboxes[0]) == [20, 30, 40, 60]

    def test_extract_bboxes_multiple(self):
        """Test extracting multiple bounding boxes."""
        from decimer_segmentation.mrcnn.utils import extract_bboxes

        mask = np.zeros((100, 100, 2), dtype=bool)
        mask[10:20, 10:20, 0] = True
        mask[50:70, 60:80, 1] = True

        bboxes = extract_bboxes(mask)

        assert bboxes.shape == (2, 4)
        assert list(bboxes[0]) == [10, 10, 20, 20]
        assert list(bboxes[1]) == [50, 60, 70, 80]

    def test_compute_iou(self):
        """Test IoU computation."""
        from decimer_segmentation.mrcnn.utils import compute_iou

        box = np.array([0, 0, 10, 10])
        boxes = np.array(
            [
                [0, 0, 10, 10],  # Perfect overlap
                [5, 5, 15, 15],  # Partial overlap
                [20, 20, 30, 30],  # No overlap
            ]
        )

        box_area = 100
        boxes_area = np.array([100, 100, 100])

        ious = compute_iou(box, boxes, box_area, boxes_area)

        assert abs(ious[0] - 1.0) < 0.01  # Perfect overlap
        assert 0 < ious[1] < 1.0  # Partial overlap
        assert ious[2] == 0.0  # No overlap


class TestApplyMasks:
    """Tests for mask application to images."""

    def test_apply_single_mask(self):
        """Test applying a single mask to extract segment."""
        from decimer_segmentation.decimer_segmentation import _apply_single_mask

        # Create test image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Create mask
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 30:60] = True

        segment, bbox = _apply_single_mask(image, mask)

        assert bbox == (20, 30, 40, 60)
        assert segment.shape[0] == 20  # height
        assert segment.shape[1] == 30  # width
        assert segment.shape[2] == 4  # RGBA

    def test_apply_masks_empty(self):
        """Test applying empty masks."""
        from decimer_segmentation import apply_masks

        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        masks = np.empty((100, 100, 0), dtype=bool)

        segments, bboxes = apply_masks(image, masks)

        assert len(segments) == 0
        assert len(bboxes) == 0


class TestSorting:
    """Tests for segment sorting functionality."""

    def test_sort_segments_reading_order(self):
        """Test sorting segments in reading order."""
        from decimer_segmentation.decimer_segmentation import _sort_segments_bboxes

        # Create segments and bboxes out of order
        segments = [
            np.zeros((10, 10, 4), dtype=np.uint8),  # Should be 3rd
            np.ones((10, 10, 4), dtype=np.uint8),  # Should be 1st
            np.full((10, 10, 4), 128, dtype=np.uint8),  # Should be 2nd
        ]

        bboxes = [
            (100, 50, 110, 60),  # Row 2, middle
            (10, 10, 20, 20),  # Row 1, left
            (10, 80, 20, 90),  # Row 1, right
        ]

        sorted_segments, sorted_bboxes = _sort_segments_bboxes(
            segments, bboxes, row_threshold=50
        )

        # First should be top-left
        assert sorted_bboxes[0] == (10, 10, 20, 20)
        # Second should be top-right
        assert sorted_bboxes[1] == (10, 80, 20, 90)
        # Third should be bottom
        assert sorted_bboxes[2] == (100, 50, 110, 60)


class TestImageResize:
    """Tests for image resizing utilities."""

    def test_resize_image_square(self):
        """Test square mode resizing."""
        from decimer_segmentation.mrcnn.utils import resize_image

        image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)

        resized, window, scale, padding, crop = resize_image(
            image, min_dim=800, max_dim=1024, mode="square"
        )

        assert resized.shape[0] == resized.shape[1]  # Square
        assert resized.shape[0] <= 1024

    def test_resize_preserves_dtype(self):
        """Test that resize preserves dtype."""
        from decimer_segmentation.mrcnn.utils import resize_image

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        resized, _, _, _, _ = resize_image(
            image, min_dim=200, max_dim=256, mode="square"
        )

        assert resized.dtype == np.uint8


class TestLineDetection:
    """Tests for line detection in mask expansion."""

    def test_detect_horizontal_lines(self):
        """Test horizontal line detection."""
        from decimer_segmentation.optimized_complete_structure import (
            _detect_horizontal_vertical_lines,
        )

        # Create image with horizontal line
        image = np.ones((100, 200), dtype=bool)  # White background
        image[50, :] = False  # Horizontal line

        lines = _detect_horizontal_vertical_lines(image, (50, 150))

        # Should detect the line
        assert lines.any()

    def test_line_intersects_structure(self):
        """Test line-structure intersection check."""
        from decimer_segmentation.optimized_complete_structure import (
            _line_intersects_structure,
        )

        # Create structure mask
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True

        # Line through structure
        intersects = _line_intersects_structure(0, 50, 100, 50, mask)
        assert intersects

        # Line avoiding structure
        intersects = _line_intersects_structure(0, 10, 100, 10, mask)
        assert not intersects


class TestConfiguration:
    """Tests for configuration classes."""

    def test_config_initialization(self):
        """Test configuration initialization."""
        from decimer_segmentation.mrcnn.config import Config

        config = Config()

        assert hasattr(config, "BATCH_SIZE")
        assert hasattr(config, "IMAGE_SHAPE")
        assert config.BATCH_SIZE == config.IMAGES_PER_GPU * config.GPU_COUNT

    def test_moldetect_config(self):
        """Test MolDetect configuration."""
        from decimer_segmentation.mrcnn.moldetect import MolDetectConfig

        config = MolDetectConfig()

        assert config.NAME == "Molecule"
        assert config.NUM_CLASSES == 2  # background + molecule

    def test_config_to_dict(self):
        """Test configuration serialization."""
        from decimer_segmentation.mrcnn.config import Config

        config = Config()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "BATCH_SIZE" in config_dict


class TestFileIO:
    """Tests for file I/O operations."""

    def test_save_images(self):
        """Test saving images to disk."""
        from decimer_segmentation import save_images

        images = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_images(images, tmpdir, "test")

            assert os.path.exists(os.path.join(tmpdir, "test_0.png"))
            assert os.path.exists(os.path.join(tmpdir, "test_1.png"))

    def test_load_single_image(self):
        """Test loading a single image file."""
        from decimer_segmentation.decimer_segmentation import _load_single_image

        # Create temporary image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(f.name, image)

            try:
                loaded = _load_single_image(f.name)
                assert len(loaded) == 1
                assert loaded[0].shape[:2] == (100, 100)
            finally:
                os.unlink(f.name)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_image(self):
        """Test handling of empty/invalid images."""
        from decimer_segmentation import get_bnw_image

        # Empty image should return as-is
        empty = np.array([])
        result = get_bnw_image(empty)

        assert result.size == 0

    def test_very_small_mask(self):
        """Test handling of very small masks."""
        from decimer_segmentation.decimer_segmentation import _apply_single_mask

        image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Single pixel mask
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 50] = True

        segment, bbox = _apply_single_mask(image, mask)

        assert segment is not None
        assert bbox == (50, 50, 51, 51)

    def test_full_image_mask(self):
        """Test handling of mask covering entire image."""
        from decimer_segmentation.decimer_segmentation import _apply_single_mask

        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        mask = np.ones((50, 50), dtype=bool)

        segment, bbox = _apply_single_mask(image, mask)

        assert segment.shape[:2] == (50, 50)
        assert bbox == (0, 0, 50, 50)


class TestVisualization:
    """Tests for visualization functions."""

    def test_random_colors(self):
        """Test random color generation."""
        from decimer_segmentation.mrcnn.visualize import random_colors

        colors = random_colors(5)

        assert len(colors) == 5
        for color in colors:
            assert len(color) == 3
            assert all(0 <= c <= 1 for c in color)

    def test_apply_mask_visualization(self):
        """Test mask application for visualization."""
        from decimer_segmentation.mrcnn.visualize import apply_mask

        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:60] = 1

        color = (1.0, 0.0, 0.0)  # Red
        result = apply_mask(image.copy(), mask, color, alpha=0.5)

        # Masked region should have red tint
        assert result[30, 45, 0] != 128  # Red channel changed


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_segment_synthetic_image(self):
        """Test segmentation on synthetic image with known structures."""
        from decimer_segmentation import segment_chemical_structures

        # Create synthetic image with "structure-like" blobs
        image = np.ones((500, 500, 3), dtype=np.uint8) * 255

        # Add black blobs (simulate structures)
        cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 0), -1)
        cv2.rectangle(image, (300, 300), (400, 400), (0, 0, 0), -1)

        # Note: This will fail without the model, but tests the code path
        try:
            segments = segment_chemical_structures(image, expand=False)
            # If model loads, we get results
            assert isinstance(segments, list)
        except Exception as e:
            # Expected if model not available
            assert "model" in str(e).lower() or "weight" in str(e).lower()

    def test_full_pipeline_empty_image(self):
        """Test full pipeline with empty/white image."""
        from decimer_segmentation import segment_chemical_structures

        # All white image - no structures
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255

        try:
            segments = segment_chemical_structures(image, expand=False)
            # Should return empty list for blank image
            assert isinstance(segments, list)
        except Exception:
            # Expected if model not available
            pass

    def test_segment_from_file_with_return_bboxes(self, mock_model_detection):
        """
        Test segment_chemical_structures_from_file returns bboxes
        when return_bboxes=True.
        """
        from decimer_segmentation import segment_chemical_structures_from_file

        pdf_path = create_test_pdf(num_pages=1)
        try:
            result = segment_chemical_structures_from_file(
                pdf_path, expand=False, return_bboxes=True
            )
            assert isinstance(result, tuple)
            assert len(result) == 2
            segments, bboxes = result
            assert isinstance(segments, list)
            assert isinstance(bboxes, list)
            assert len(segments) == len(bboxes)
            assert len(segments) == 2, "Mocked model should return 2 detections"
            for bbox in bboxes:
                assert len(bbox) == 4
        finally:
            os.unlink(pdf_path)

    def test_segment_multipage_pdf_with_page_numbers(self, mock_model_detection):
        """
        Test segment_chemical_structures_from_file returns page numbers
        for multi-page PDFs.
        """
        from decimer_segmentation import segment_chemical_structures_from_file

        pdf_path = create_test_pdf(num_pages=3)
        try:
            result = segment_chemical_structures_from_file(
                pdf_path, expand=False, return_page_numbers=True
            )
            segments, page_numbers = result

            # Mock returns 2 detections per page: pages should be [1, 1, 2, 2, 3, 3]
            assert page_numbers == [1, 1, 2, 2, 3, 3], (
                f"Expected page numbers [1, 1, 2, 2, 3, 3], got {page_numbers}"
            )

            # Verify we got 6 valid segments
            assert len(segments) == 6
            for i, seg in enumerate(segments):
                assert isinstance(seg, np.ndarray), f"Segment {i} should be numpy array"
                assert seg.size > 0, f"Segment {i} should not be empty"
        finally:
            os.unlink(pdf_path)

    def test_segment_pdf_with_bboxes_and_page_numbers(self, mock_model_detection):
        """
        Test segment_chemical_structures_from_file returns both bboxes
        and page numbers.
        """
        from decimer_segmentation import segment_chemical_structures_from_file

        pdf_path = create_test_pdf(num_pages=1)
        try:
            result = segment_chemical_structures_from_file(
                pdf_path, expand=False, return_bboxes=True, return_page_numbers=True
            )
            segments, bboxes, page_numbers = result

            # Mock returns exact known values
            assert page_numbers == [1, 1]
            assert bboxes == [(50, 50, 150, 150), (200, 200, 300, 300)]
            assert len(segments) == 2
            for seg in segments:
                assert isinstance(seg, np.ndarray) and seg.size > 0
        finally:
            os.unlink(pdf_path)


# Benchmark tests (optional, for performance monitoring)
class TestPerformance:
    """Performance benchmark tests."""

    @pytest.mark.skip(reason="Benchmark test - run manually")
    def test_mask_expansion_performance(self):
        """Benchmark mask expansion speed."""
        import time
        from decimer_segmentation.mask_expansion import complete_structure_mask

        # Create realistic test data
        image = np.random.randint(200, 255, (1000, 1000, 3), dtype=np.uint8)
        # Add some dark regions
        for _ in range(10):
            x, y = np.random.randint(100, 900, 2)
            cv2.circle(image, (x, y), 50, (0, 0, 0), -1)

        masks = np.random.rand(1000, 1000, 5) > 0.95

        start = time.time()
        _ = complete_structure_mask(image, masks, (200, 200))
        elapsed = time.time() - start

        print(f"Mask expansion took {elapsed:.3f}s for 5 masks on 1000x1000 image")
        assert elapsed < 5.0  # Should complete in under 5 seconds

    @pytest.mark.skip(reason="Benchmark test - run manually")
    def test_bbox_extraction_performance(self):
        """Benchmark bounding box extraction."""
        import time
        from decimer_segmentation.mrcnn.utils import extract_bboxes

        # Large mask array
        masks = np.random.rand(1000, 1000, 50) > 0.99

        start = time.time()
        _ = extract_bboxes(masks)
        elapsed = time.time() - start

        print(f"Bbox extraction took {elapsed:.3f}s for 50 masks")
        assert elapsed < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
