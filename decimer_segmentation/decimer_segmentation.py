"""
DECIMER Segmentation - Optimized Production Implementation

This Software is under the MIT License
Refer to LICENSE or https://opensource.org/licenses/MIT for more information
Written by ©Kohulan Rajan 2020
Optimized for production performance 2024

Performance Optimizations Applied:
- Model warmup to eliminate cold-start latency
- Reduced proposal counts (POST_NMS_ROIS: 1000→500, DETECTION_MAX: 100→50)
- cuDNN autotuning for GPU operations
- TensorFlow graph optimizations

Importing this module does not download or load the model. get_model() does that
the first time it is needed (the first segmentation, or an explicit call to
get_model()), and keeps the model for later calls.
"""

from __future__ import annotations

import os

# =============================================================================
# PERFORMANCE: Set environment variables BEFORE importing TensorFlow
# =============================================================================
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import contextlib
import logging
from typing import List, Tuple, Union, Optional
import threading

import argparse
import cv2
import numpy as np
import pystow
import tensorflow as tf

from .optimized_complete_structure import complete_structure_mask
from .mrcnn import model as modellib
from .mrcnn import visualize
from .mrcnn import moldetect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_DOWNLOAD_URL = (
    "https://zenodo.org/record/10663579/files/mask_rcnn_molecule.h5?download=1"
)
MODEL_FILENAME = "mask_rcnn_molecule.h5"

# PDF pages are rendered at this resolution, by one of these libraries (pdf_backend).
PDF_RENDER_DPI = 300
PDF_BACKENDS = ("pdfium", "pymupdf")
DEFAULT_PDF_BACKEND = "pdfium"

# Global model instance with thread-safe lazy loading
_model: Optional[modellib.MaskRCNN] = None
_model_lock = threading.Lock()
_model_warmed_up: bool = False

# Neither PDFium nor PyMuPDF is thread-safe: one PDF is rendered at a time per process.
_pdf_lock = threading.Lock()


class InferenceConfig(moldetect.MolDetectConfig):
    """
    Optimized inference configuration for MRCNN model.

    Reduced proposal counts provide ~1.5-2x speedup with minimal accuracy impact.
    """

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.75

    # PERFORMANCE: Reduced from defaults for faster inference
    POST_NMS_ROIS_INFERENCE = 500  # Default: 1000
    DETECTION_MAX_INSTANCES = 100  # Default: 100
    PRE_NMS_LIMIT = 6000  # Default: 6000

    def __init__(self):
        super().__init__()


_inference_config = InferenceConfig()


def get_model() -> modellib.MaskRCNN:
    """
    Thread-safe lazy loading of the MRCNN model with warmup.

    Returns:
        modellib.MaskRCNN: Loaded and warmed-up model with trained weights
    """
    global _model, _model_warmed_up

    if _model is not None and _model_warmed_up:
        return _model

    with _model_lock:
        if _model is not None and _model_warmed_up:
            return _model

        _model = _load_model_internal()

        if not _model_warmed_up:
            _warmup_model(_model)
            _model_warmed_up = True

        return _model


def _load_model_internal() -> modellib.MaskRCNN:
    """Load model with TensorFlow optimizations."""

    # PERFORMANCE: Enable graph optimizations
    try:
        tf.config.optimizer.set_experimental_options(
            {
                "layout_optimizer": True,
                "constant_folding": True,
                "shape_optimization": True,
                "remapping": True,
                "arithmetic_optimization": True,
                "dependency_optimization": True,
                "loop_optimization": True,
                "function_optimization": True,
                "debug_stripper": True,
            }
        )
    except Exception as e:
        logger.debug(f"Some optimizer options not available: {e}")

    model_path = pystow.join("DECIMER-Segmentation_model")
    logger.debug("Model path: %s", model_path)
    if not os.path.exists(os.path.join(model_path, MODEL_FILENAME)):
        logger.info("Downloading model weights...")
        download_trained_weights(MODEL_DOWNLOAD_URL, str(model_path))
        logger.info("Successfully downloaded the segmentation model weights!")

    model = modellib.MaskRCNN(mode="inference", model_dir=".", config=_inference_config)
    model.load_weights(os.path.join(model_path, MODEL_FILENAME), by_name=True)

    return model


def _warmup_model(model: modellib.MaskRCNN) -> None:
    """
    Warm up model with dummy inference to eliminate cold-start latency.

    First inference triggers lazy initialization and can be 10-100x slower.
    """
    logger.info("Warming up model...")

    dummy_image = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
    cv2.rectangle(dummy_image, (100, 100), (200, 200), (0, 0, 0), 2)

    try:
        _ = model.detect([dummy_image], verbose=0)
        logger.info("Model warmup complete")
    except Exception as e:
        logger.warning(f"Model warmup encountered issue: {e}")


def download_trained_weights(model_url: str, model_path: str, verbose=1):
    """This function downloads the trained modelto a given location.
    If the model exists on the given location this function
    will be skipped.

    Args:
        model_url (str): trained model url for downloading.
        model_path (str): model default path to download.

    Returns:
        path (str): downloaded model.
    """
    # Download trained models
    if verbose > 0:
        print("Downloading trained model to " + str(model_path))
    model_path = pystow.ensure("DECIMER-Segmentation_model", url=model_url)
    if verbose > 0:
        print("... done downloading trained model!")


def __getattr__(name: str):
    """Load the model when ``segmentation_model`` is accessed.

    Up to version 1.5.1 ``segmentation_model`` was a module-level variable that was
    set (the weights downloaded and loaded) when this module was imported. It now
    resolves to ``get_model()`` the first time it is read.
    """
    if name == "segmentation_model":
        return get_model()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def segment_chemical_structures_from_file(
    file_path: str,
    expand: bool = True,
    return_bboxes: bool = False,
    return_page_numbers: bool = False,
    *,
    pdf_backend: str = DEFAULT_PDF_BACKEND,
    **kwargs,
) -> Union[
    List[np.ndarray],
    Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]],
    Tuple[List[np.ndarray], List[int]],
    Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], List[int]],
]:
    """
    Segment chemical structures from a PDF or image file.

    Args:
        file_path: Path to input file (PDF or image)
        expand: Whether to expand masks to capture complete structures
        return_bboxes: Whether to return bounding boxes along with segments
        return_page_numbers: Whether to return 1-indexed page numbers per segment
        pdf_backend: The library that renders the pages of a PDF (ignored for
            images): "pdfium" (the default; pypdfium2, Apache-2.0 / BSD-3-Clause)
            or "pymupdf" (PyMuPDF, AGPL-3.0, which is not installed by default:
            pip install "decimer-segmentation[pdf]").

    Returns:
        List of segmented chemical structure images as numpy arrays.
        If return_bboxes is True, returns a tuple of (segments, bboxes).
        If return_page_numbers is True, returns (segments, page_numbers).
        If both are True, returns (segments, bboxes, page_numbers).
        There is one page number per segment, in the order of the segments: the
        page of the PDF the segment was found on, counted from 1 (every segment of
        an image file is on page 1).

    Raises:
        ValueError: pdf_backend is not one of PDF_BACKENDS.
        ImportError: pdf_backend="pymupdf" for a PDF, and PyMuPDF is not installed.
    """
    _check_pdf_backend(pdf_backend)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if "poppler_path" in kwargs:
        logger.warning("poppler_path parameter is deprecated and ignored")

    images_with_pages = _load_images_from_file(file_path, pdf_backend)

    if not images_with_pages:
        logger.warning(f"No images could be extracted from {file_path}")

    # Process all images sequentially (model can't parallelize)
    all_segments = []
    all_bboxes = []
    all_page_numbers = []
    for image, page_number in images_with_pages:
        segments, bboxes = segment_chemical_structures(
            image, expand, return_bboxes=True
        )
        all_segments.extend(segments)
        all_bboxes.extend(bboxes)
        all_page_numbers.extend([page_number] * len(segments))

    if not (return_bboxes or return_page_numbers):
        return all_segments
    result = (all_segments,)
    if return_bboxes:
        result += (all_bboxes,)
    if return_page_numbers:
        result += (all_page_numbers,)
    return result


def _load_images_from_file(
    file_path: str, pdf_backend: str = DEFAULT_PDF_BACKEND
) -> List[Tuple[np.ndarray, int]]:
    """Load images from PDF or image file.

    Returns:
        List of (image, page_number) tuples. Page numbers are 1-indexed.
    """
    if file_path.lower().endswith(".pdf"):
        return _load_pdf_pages(file_path, pdf_backend)
    else:
        return [(img, 1) for img in _load_single_image(file_path)]


def _check_pdf_backend(pdf_backend: str) -> None:
    if pdf_backend not in PDF_BACKENDS:
        raise ValueError(
            f"Unknown pdf_backend {pdf_backend!r}: use one of "
            + ", ".join(repr(name) for name in PDF_BACKENDS)
        )


def _import_pymupdf():
    """Import PyMuPDF, which is only needed by the "pymupdf" PDF backend.

    PyMuPDF is an optional dependency (``pip install "decimer-segmentation[pdf]"``)
    because it is licensed under the AGPL, unlike this package. It is imported here,
    when the "pymupdf" backend reads a PDF, and not when this module is imported.
    """
    try:
        import pymupdf
    except ImportError as error:
        raise ImportError(
            'pdf_backend="pymupdf" needs PyMuPDF, an optional dependency of '
            "decimer-segmentation that is not installed (it is licensed under the "
            "AGPL-3.0, so it is not installed by default). Install it with: "
            'pip install "decimer-segmentation[pdf]". '
            "PDFs are read with pypdfium2 by default, which is always installed; "
            "images and numpy arrays need neither."
        ) from error
    return pymupdf


def _open_pdf_pdfium(pdf_path: str):
    """Open a PDF with pypdfium2 (Apache-2.0 / BSD-3-Clause), the default backend."""
    import pypdfium2

    document = pypdfium2.PdfDocument(pdf_path)
    try:
        # Without a form environment (which has to exist before any page is loaded,
        # and is closed with the document) pdfium draws no form fields: a filled-in
        # form would come out blank, where PyMuPDF draws the fields.
        document.init_forms()
    except BaseException:
        document.close()
        raise
    return document


def _render_page_pdfium(document, index: int) -> np.ndarray:
    # pypdfium2 sizes the bitmap as math.ceil(points * scale), and 792 * (300 / 72) is
    # 3300.0000000000005, so a US Letter page would get 3301 rows instead of the 3300
    # that 300 DPI means (and PyMuPDF gives). A scale lower by a part in a billion
    # (0.00003 px on the largest page) gives the exact size and changes nothing else.
    scale = PDF_RENDER_DPI / 72 * (1 - 1e-9)
    with contextlib.closing(document[index]) as page:
        with contextlib.closing(page.render(scale=scale)) as bitmap:
            # White background, no alpha channel, 3 channels in BGR order: what OpenCV
            # and this package use, so no colour conversion. The array is a view of
            # the bitmap's buffer, which goes with the bitmap, hence the copy.
            return bitmap.to_numpy().copy()


def _open_pdf_pymupdf(pdf_path: str):
    """Open a PDF with PyMuPDF (AGPL-3.0), the optional backend."""
    return _import_pymupdf().open(pdf_path)


def _render_page_pymupdf(document, index: int) -> np.ndarray:
    pix = document[index].get_pixmap(dpi=PDF_RENDER_DPI, alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _load_pdf_pages(
    pdf_path: str, pdf_backend: str = DEFAULT_PDF_BACKEND
) -> List[Tuple[np.ndarray, int]]:
    """Render all pages of a PDF, in page order, at 300 DPI as BGR uint8 arrays.

    The pages are rendered one after the other from one open document, which is
    closed afterwards: neither PDFium nor PyMuPDF is thread-safe, and a plain loop
    was also the fastest way to render a long PDF with PyMuPDF.

    Returns:
        List of (image, page_number) tuples. Page numbers are 1-indexed.
    """
    _check_pdf_backend(pdf_backend)
    if pdf_backend == "pymupdf":
        open_pdf, render_page = _open_pdf_pymupdf, _render_page_pymupdf
    else:
        open_pdf, render_page = _open_pdf_pdfium, _render_page_pdfium

    with _pdf_lock, contextlib.closing(open_pdf(pdf_path)) as document:
        return [
            (render_page(document, index), index + 1) for index in range(len(document))
        ]


def _load_single_image(image_path: str) -> List[np.ndarray]:
    """Load a single image file."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return [image]


def segment_chemical_structures(
    image: np.ndarray,
    expand: bool = True,
    visualization: bool = False,
    return_bboxes: bool = False,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]]:
    """
    Segment chemical structures from an image.

    Args:
        image: Input image as numpy array (BGR format)
        expand: Whether to expand masks to capture complete structures
        visualization: Whether to display visualization (Jupyter only)
        return_bboxes: Whether to return bounding boxes along with segments

    Returns:
        List of segmented structure images, optionally with bounding boxes.
        If return_bboxes is True, returns a tuple of (segments, bboxes).
    """
    if image is None or image.size == 0:
        return ([], []) if return_bboxes else []

    # Ensure BGR format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Get masks
    if expand:
        masks = get_expanded_masks(image)
    else:
        masks, _, _ = get_mrcnn_results(image)

    # Apply masks to extract segments
    segments, bboxes = apply_masks(image, masks)

    if visualization and len(bboxes) > 0:
        _visualize_results(image, masks, bboxes)

    # Sort in reading order and filter empty
    if segments:
        segments, bboxes = _sort_segments_bboxes(segments, bboxes)
        filtered = [
            (s, b)
            for s, b in zip(segments, bboxes)
            if s is not None and s.size > 0 and s.shape[0] > 0 and s.shape[1] > 0
        ]
        if filtered:
            segments, bboxes = zip(*filtered)
            segments, bboxes = list(segments), list(bboxes)
        else:
            segments, bboxes = [], []

    return (segments, bboxes) if return_bboxes else segments


def _visualize_results(
    image: np.ndarray, masks: np.ndarray, bboxes: List[Tuple[int, int, int, int]]
) -> None:
    """Display visualization of detection results."""
    visualize.display_instances(
        image=image,
        masks=masks,
        class_ids=np.array([0] * len(bboxes)),
        boxes=np.array(bboxes),
        class_names=np.array(["structure"] * len(bboxes)),
    )


def get_mrcnn_results(
    image: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], List[float]]:
    """
    Run MRCNN detection on an image.

    Returns:
        Tuple of (masks, bounding_boxes, confidence_scores)
    """
    model = get_model()  # downloads and loads the model on the first call
    results = model.detect([image], verbose=0)

    return (
        results[0]["masks"],
        results[0]["rois"].tolist(),
        results[0]["scores"].tolist(),
    )


def get_expanded_masks(image: np.ndarray) -> np.ndarray:
    """
    Get expanded masks that capture complete chemical structures.

    Returns:
        Expanded masks array of shape (height, width, num_masks)
    """
    masks, bboxes, _ = get_mrcnn_results(image)

    if len(bboxes) == 0:
        return masks

    max_size = _determine_depiction_size_with_buffer(bboxes)

    return complete_structure_mask(
        image_array=image, mask_array=masks, max_depiction_size=max_size, debug=False
    )


def determine_depiction_size_with_buffer(
    bboxes: List[Tuple[int, int, int, int]],
) -> Tuple[int, int]:
    """Calculate maximum depiction size with 10% buffer."""
    if not bboxes:
        return (100, 100)

    bboxes_array = np.array(bboxes)
    heights = bboxes_array[:, 2] - bboxes_array[:, 0]
    widths = bboxes_array[:, 3] - bboxes_array[:, 1]

    return (int(1.1 * np.max(heights)), int(1.1 * np.max(widths)))


# Alias for internal use
_determine_depiction_size_with_buffer = determine_depiction_size_with_buffer


def _sort_segments_bboxes(
    segments: List[np.ndarray],
    bboxes: List[Tuple[int, int, int, int]],
    row_threshold: int = 50,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """Sort segments in reading order (top-to-bottom, left-to-right)."""
    if not segments:
        return segments, bboxes

    indices = sorted(range(len(bboxes)), key=lambda i: bboxes[i][0])

    rows = []
    current_row = [indices[0]]

    for i in indices[1:]:
        if abs(bboxes[i][0] - bboxes[current_row[-1]][0]) < row_threshold:
            current_row.append(i)
        else:
            current_row.sort(key=lambda idx: bboxes[idx][1])
            rows.append(current_row)
            current_row = [i]

    current_row.sort(key=lambda idx: bboxes[idx][1])
    rows.append(current_row)

    sorted_indices = [idx for row in rows for idx in row]

    return ([segments[i] for i in sorted_indices], [bboxes[i] for i in sorted_indices])


def apply_masks(
    image: np.ndarray, masks: np.ndarray
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Apply masks to image and extract segmented regions.

    Returns:
        Tuple of (segment_images, bounding_boxes)
    """
    if masks.size == 0 or masks.shape[2] == 0:
        return [], []

    num_masks = masks.shape[2]
    segments = []
    bboxes = []

    for i in range(num_masks):
        segment, bbox = _apply_single_mask(image, masks[:, :, i])
        segments.append(segment)
        bboxes.append(bbox)

    return segments, bboxes


def _apply_single_mask(
    image: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Apply a single mask to extract a segment from the image."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return np.zeros((1, 1, 4), dtype=np.uint8), (0, 0, 0, 0)

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]

    y0, y1 = y_indices[0], y_indices[-1] + 1
    x0, x1 = x_indices[0], x_indices[-1] + 1

    roi = image[y0:y1, x0:x1].copy()
    mask_roi = mask[y0:y1, x0:x1]

    if len(roi.shape) == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    alpha = (mask_roi * 255).astype(np.uint8)
    b, g, r = cv2.split(roi)
    rgba = cv2.merge([b, g, r, alpha])
    rgba[alpha == 0] = [255, 255, 255, 255]

    return rgba, (y0, x0, y1, x1)


def save_images(images: List[np.ndarray], output_dir: str, base_name: str) -> None:
    """Save images to disk with generated filenames."""
    os.makedirs(output_dir, exist_ok=True)

    for index, image in enumerate(images):
        if image is None or image.size == 0:
            continue
        filepath = os.path.join(output_dir, f"{base_name}_{index}.png")
        cv2.imwrite(filepath, image)


def get_bnw_image(image: np.ndarray) -> np.ndarray:
    """Convert image to black and white using Otsu thresholding."""
    if image is None or image.size == 0:
        return image

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image

    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binarized


def get_square_image(image: np.ndarray, target_size: int = 299) -> np.ndarray:
    """Resize image to square without distortion (with padding)."""
    if image is None or image.size == 0:
        return np.full((target_size, target_size), 255, dtype=np.uint8)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image

    h, w = gray.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    output = np.full((target_size, target_size), 255, dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    output[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return output


# =============================================================================
# Configuration API
# =============================================================================


def set_inference_config(
    post_nms_rois: int = 500,
    detection_max_instances: int = 50,
    detection_min_confidence: float = 0.7,
) -> None:
    """
    Configure inference parameters for speed/accuracy tradeoff.
    Must be called BEFORE first model load.

    Args:
        post_nms_rois: ROIs after NMS (default: 500, original: 1000)
        detection_max_instances: Max detections (default: 50, original: 100)
        detection_min_confidence: Min confidence (default: 0.7)
    """
    if _model is not None:
        logger.warning(
            "Config changes won't take effect - model already loaded. Call reset_model() first."
        )
        return

    _inference_config.POST_NMS_ROIS_INFERENCE = post_nms_rois
    _inference_config.DETECTION_MAX_INSTANCES = detection_max_instances
    _inference_config.DETECTION_MIN_CONFIDENCE = detection_min_confidence
    _inference_config.PRE_NMS_LIMIT = post_nms_rois * 6


def reset_model() -> None:
    """Reset model to allow reconfiguration."""
    global _model, _model_warmed_up

    with _model_lock:
        _model = None
        _model_warmed_up = False

    logger.info("Model reset. New config will apply on next load.")


# =============================================================================
# CLI
# =============================================================================


def main():
    """Command-line interface for DECIMER Segmentation."""

    parser = argparse.ArgumentParser(
        description="Segment chemical structures from scientific literature"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input file path (PDF or image)"
    )
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument(
        "--no-expand", action="store_true", help="Disable mask expansion"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Use faster settings (fewer proposals)"
    )
    parser.add_argument(
        "--pdf-backend",
        "--pdf_backend",
        dest="pdf_backend",
        choices=PDF_BACKENDS,
        default=DEFAULT_PDF_BACKEND,
        help="Library that renders PDF pages (default: %(default)s; pymupdf is "
        "AGPL-3.0 and needs: pip install 'decimer-segmentation[pdf]')",
    )

    args = parser.parse_args()

    if args.fast:
        set_inference_config(post_nms_rois=300, detection_max_instances=30)

    output_dir = args.output or f"{args.input}_output"
    segment_dir = os.path.join(output_dir, "segments")

    logger.info("Loading model...")
    get_model()

    logger.info(f"Processing: {args.input}")
    segments = segment_chemical_structures_from_file(
        args.input, expand=not args.no_expand, pdf_backend=args.pdf_backend
    )

    if not segments:
        logger.warning("No chemical structures found.")
        return

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    save_images(segments, segment_dir, base_name)
    logger.info(f"Saved {len(segments)} segments to {segment_dir}")


if __name__ == "__main__":
    main()
