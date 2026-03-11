"""
Pytest Configuration and Fixtures

Shared fixtures for DECIMER Segmentation tests.
"""

import os
import sys
import warnings
import pytest
import numpy as np
import pystow
import tempfile

# Suppress SWIG-related deprecation warnings from third-party libraries (e.g., OpenCV)
# These warnings come from importlib._bootstrap and cannot be easily fixed
warnings.filterwarnings(
    "ignore",
    message=r"builtin type Swig.*",
    category=DeprecationWarning,
    module="importlib._bootstrap",
)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type swigvarlink.*",
    category=DeprecationWarning,
    module="importlib._bootstrap",
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_image():
    """Generate a sample test image."""
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255
    return image


@pytest.fixture
def sample_image_with_structures():
    """Generate a sample image with structure-like shapes."""
    import cv2

    image = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # Add some black shapes
    cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 0), 2)
    cv2.circle(image, (300, 100), 40, (0, 0, 0), 2)
    cv2.line(image, (100, 100), (120, 120), (0, 0, 0), 2)

    # Add filled structure in bottom
    cv2.rectangle(image, (200, 300), (350, 450), (0, 0, 0), -1)

    return image


@pytest.fixture
def sample_mask():
    """Generate a sample binary mask."""
    mask = np.zeros((500, 500), dtype=bool)
    mask[100:200, 100:200] = True
    return mask


@pytest.fixture
def sample_masks():
    """Generate multiple sample masks."""
    masks = np.zeros((500, 500, 3), dtype=bool)
    masks[50:100, 50:100, 0] = True
    masks[200:300, 200:300, 1] = True
    masks[350:450, 100:200, 2] = True
    return masks


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_image_file(sample_image):
    """Create a temporary image file."""
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, sample_image)
        yield f.name
        os.unlink(f.name)


@pytest.fixture
def mock_model_detection(monkeypatch):
    """
    Mock the model detection to return fake results without running the actual model.
    Returns 2 fake detections per image for testing.
    """

    def mock_get_mrcnn_results(image):
        """Return fake detection results."""
        h, w = image.shape[:2]
        # Create 2 fake masks
        masks = np.zeros((h, w, 2), dtype=bool)
        masks[50:150, 50:150, 0] = True
        masks[200:300, 200:300, 1] = True

        # Fake bounding boxes (y0, x0, y1, x1)
        bboxes = [(50, 50, 150, 150), (200, 200, 300, 300)]

        # Fake confidence scores
        scores = [0.95, 0.92]

        return masks, bboxes, scores

    # Patch the get_mrcnn_results function
    import decimer_segmentation.decimer_segmentation as ds

    monkeypatch.setattr(ds, "get_mrcnn_results", mock_get_mrcnn_results)


@pytest.fixture(scope="session")
def model_available():
    """Check if the model weights are available."""
    model_path = os.path.join(
        str(pystow.join("DECIMER-Segmentation_model")), "mask_rcnn_molecule.h5"
    )
    return os.path.exists(model_path)


# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_model: marks tests that require model weights"
    )
    config.addinivalue_line("markers", "benchmark: marks benchmark tests")


# Skip model-dependent tests if model not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available resources."""
    model_path = os.path.join(
        str(pystow.join("DECIMER-Segmentation_model")), "mask_rcnn_molecule.h5"
    )
    model_available = os.path.exists(model_path)

    if not model_available:
        skip_model = pytest.mark.skip(reason="Model weights not available")
        for item in items:
            if "requires_model" in item.keywords:
                item.add_marker(skip_model)
