"""
* This Software is under the MIT License
* Refer to LICENSE or https://opensource.org/licenses/MIT for more information
* Written by ©Kohulan Rajan 2020
* Optimized for performance
"""

import os
import requests
import cv2
import argparse
import numpy as np
from multiprocessing import Pool
import fitz  # PyMuPDF
from typing import List, Tuple, Union
from PIL import Image
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from .optimized_complete_structure import complete_structure_mask
from .mrcnn import model as modellib
from .mrcnn import visualize
from .mrcnn import moldetect

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))

# Global model instance (lazy loading)
_model = None


class InferenceConfig(moldetect.MolDetectConfig):
    """
    Inference configuration class for MRCNN
    """

    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7


def segment_chemical_structures_from_file(
    file_path: str, expand: bool = True
) -> List[np.array]:
    """
    This function runs the segmentation model as well as the mask expansion
    on a pdf document or an image of a page from a scientific publication.
    It returns a list of segmented chemical structure depictions (np.array)

    Args:
        file_path (str): image of a page from a scientific publication
        expand (bool): indicates whether or not to use mask expansion
        poppler_path: Deprecated parameter - no longer needed with PyMuPDF

    Returns:
        List[np.array]: expanded segments (shape: (h, w, num_masks))
    """
    if file_path[-3:].lower() == "pdf":
        # Convert PDF to images using PyMuPDF with optimized settings
        pdf_document = fitz.open(file_path)
        images = []

        # Pre-allocate list for known size
        images = [None] * pdf_document.page_count

        # Use thread pool for parallel page rendering
        def render_page(page_num):
            page = pdf_document[page_num]
            # Render page to image with 300 DPI
            matrix = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=matrix, alpha=False)  # Skip alpha channel
            # Direct conversion to numpy array
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.h, pix.w, pix.n
            )
            return page_num, img_array

        # Use thread pool for I/O bound operations
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(render_page, i) for i in range(pdf_document.page_count)
            ]
            for future in futures:
                page_num, img_array = future.result()
                images[page_num] = img_array

        pdf_document.close()

        # Filter out any None values
        images = [img for img in images if img is not None]
    else:
        # Use faster image reading with proper flags
        images = [cv2.imread(file_path, cv2.IMREAD_COLOR)]

    if len(images) > 1:
        # Use optimized multiprocessing
        with Pool(min(4, len(images))) as pool:
            starmap_args = [(im, expand) for im in images]
            segments = pool.starmap(segment_chemical_structures, starmap_args)
            # More efficient flattening
            segments = [seg for sublist in segments for seg in sublist]
    else:
        segments = segment_chemical_structures(images[0], expand)

    return segments


def segment_chemical_structures(
    image: np.array,
    expand: bool = True,
    visualization: bool = False,
    return_bboxes: bool = False,
) -> Union[List[np.array], Tuple[List[np.array], List[Tuple[int, int, int, int]]]]:
    """
    This function runs the segmentation model as well as the mask expansion
    -> returns a List of segmented chemical structure depictions (np.array)

    Args:
        image (np.array): image of a page from a scientific publication
        expand (bool): indicates whether or not to use mask expansion
        visualization (bool): indicates whether or not to visualize the
                                results (only works in Jupyter notebook)
        return_bboxes (bool): indicates whether to return bounding boxes along with segments

    Returns:
        If return_bboxes is False:
            List[np.array]: expanded segments sorted in top->bottom, left->right order
        If return_bboxes is True:
            Tuple[List[np.array], List[Tuple]]: segments and bounding boxes
    """
    if not expand:
        masks, bboxes, _ = get_mrcnn_results(image)
    else:
        masks = get_expanded_masks(image)

    segments, bboxes = apply_masks(image, masks)

    if visualization:
        visualize.display_instances(
            image=image,
            masks=masks,
            class_ids=np.array([0] * len(bboxes)),
            boxes=np.array(bboxes),
            class_names=np.array(["structure"] * len(bboxes)),
        )

    if len(segments) > 0:
        segments, bboxes = sort_segments_bboxes(segments, bboxes)

    # Vectorized filtering for valid segments
    segments = [
        segment for segment in segments if segment.shape[0] > 0 and segment.shape[1] > 0
    ]

    if return_bboxes:
        return segments, bboxes
    else:
        return segments


def determine_depiction_size_with_buffer(
    bboxes: List[Tuple[int, int, int, int]],
) -> Tuple[int, int]:
    """
    This function takes a list of bounding boxes and returns 1.1 * the maximal
    depiction size (height, width) of the depicted chemical structures.

    Args:
        bboxes (List[Tuple[int, int, int, int]]): bounding boxes of the structure
            depictions (y0, x0, y1, x1)

    Returns:
        Tuple [int, int]: average depiction size (height, width)
    """
    # Vectorized computation for better performance
    bboxes_array = np.array(bboxes)
    heights = bboxes_array[:, 2] - bboxes_array[:, 0]
    widths = bboxes_array[:, 3] - bboxes_array[:, 1]

    height = int(1.1 * np.max(heights))
    width = int(1.1 * np.max(widths))
    return height, width


def sort_segments_bboxes(
    segments: List[np.array],
    bboxes: List[Tuple[int, int, int, int]],  # (y0, x0, y1, x1)
    same_row_pixel_threshold=50,
) -> Tuple[List[np.array], List[Tuple[int, int, int, int]]]:
    """
    Sorts segments and bounding boxes in "reading order"

    Args:
        segments - image segments to be sorted
        bboxes - bounding boxes containing edge coordinates of the image segments
        same_row_pixel_threshold - how many pixels apart can two pixels be to be
            considered "on the same row"

    Returns:
        segments and bboxes in reading order
    """
    # Create index array for efficient sorting
    indices = list(range(len(bboxes)))

    # Sort indices by y-coordinate
    indices.sort(key=lambda i: bboxes[i][0])

    # Group bounding boxes by rows
    rows = []
    current_row = [indices[0]]

    for i in indices[1:]:
        if abs(bboxes[i][0] - bboxes[current_row[-1]][0]) < same_row_pixel_threshold:
            current_row.append(i)
        else:
            # Sort current row by x-coordinate
            current_row.sort(key=lambda idx: bboxes[idx][1])
            rows.append(current_row)
            current_row = [i]

    # Don't forget the last row
    current_row.sort(key=lambda idx: bboxes[idx][1])
    rows.append(current_row)

    # Flatten the sorted indices
    sorted_indices = [idx for row in rows for idx in row]

    # Apply sorting to segments and bboxes
    sorted_segments = [segments[i] for i in sorted_indices]
    sorted_bboxes = [bboxes[i] for i in sorted_indices]

    return sorted_segments, sorted_bboxes


@lru_cache(maxsize=1)
def load_model() -> modellib.MaskRCNN:
    """
    This function loads the segmentation model and returns it. The weights
    are downloaded if necessary. Cached to avoid reloading.

    Returns:
        modellib.MaskRCNN: MRCNN model with trained weights
    """
    # Define directory with trained model weights
    root_dir = os.path.split(__file__)[0]
    model_path = os.path.join(root_dir, "mask_rcnn_molecule.h5")

    # Download trained weights if needed
    if not os.path.exists(model_path):
        print("Downloading model weights...")
        url = (
            "https://zenodo.org/record/10663579/files/mask_rcnn_molecule.h5?download=1"
        )
        # Use streaming download for large files
        with requests.get(url, stream=True) as req:
            req.raise_for_status()
            with open(model_path, "wb") as model_file:
                for chunk in req.iter_content(chunk_size=8192):
                    model_file.write(chunk)
        print("Successfully downloaded the segmentation model weights!")

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=".", config=InferenceConfig())
    # Load weights
    model.load_weights(model_path, by_name=True)
    return model


def get_expanded_masks(image: np.array) -> np.array:
    """
    This function runs the segmentation model and returns an
    array with the masks (shape: height, width, num_masks).
    Slicing along the third axis of the output of this function
    yields a binary array of shape (h, w) for a single structure.

    Args:
        image (np.array): image of a page from a scientific publication

    Returns:
        np.array: expanded masks (shape: (h, w, num_masks))
    """
    # Structure detection with MRCNN
    masks, bboxes, _ = get_mrcnn_results(image)
    if len(bboxes) == 0:
        return masks

    size = determine_depiction_size_with_buffer(bboxes)
    # Mask expansion
    expanded_masks = complete_structure_mask(
        image_array=image, mask_array=masks, max_depiction_size=size, debug=False
    )
    return expanded_masks


def get_mrcnn_results(
    image: np.array,
) -> Tuple[np.array, List[Tuple[int]], List[float]]:
    """
    This function runs the segmentation model as well as the mask
    expansion mechanism and returns an array with the masks (shape:
    height, width, num_masks), a list of bounding boxes and a list
    of confidence scores.
    Slicing along the third axis of the mask output of this function
    yields a binary array of shape (h, w) for a single structure.

    Args:
        image (np.array): image of a page from a scientific publication
    Returns:
        np.array: expanded masks (shape: (h, w, num_masks))
        List[Tuple[int]]: bounding boxes [(y0, x0, y1, x1), ...]
        List[float]: confidence scores
    """
    # Ensure model is loaded
    model = get_model()

    results = model.detect([image], verbose=1)
    scores = results[0]["scores"]
    bboxes = results[0]["rois"]
    masks = results[0]["masks"]
    return masks, bboxes, scores


def apply_masks(
    image: np.array, masks: np.array
) -> Tuple[List[np.array], List[Tuple[int, int, int, int]]]:
    """
    This function takes an image and the masks for this image
    (shape: (h, w, num_structures)) and returns a list of segmented
    chemical structure depictions (np.array) and their bounding boxes

    Args:
        image (np.array): image of a page from a scientific publication
        masks (np.array): masks (shape: (h, w, num_masks))

    Returns:
        List[np.array]: segmented chemical structure depictions
        List[Tuple[int, int, int, int]]: bounding boxes for each segment (y0, x0, y1, x1)
    """
    if masks.shape[2] == 0:
        return [], []

    # Pre-allocate lists for better performance
    num_masks = masks.shape[2]
    segmented_images = [None] * num_masks
    bboxes = [None] * num_masks

    # Process masks in parallel for better performance
    def process_mask(i):
        mask = masks[:, :, i]
        return apply_mask(image, mask)

    # Use thread pool for I/O bound operations
    with ThreadPoolExecutor(max_workers=min(4, num_masks)) as executor:
        futures = [executor.submit(process_mask, i) for i in range(num_masks)]
        for i, future in enumerate(futures):
            segmented_images[i], bboxes[i] = future.result()

    return segmented_images, bboxes


def apply_mask(
    image: np.array, mask: np.array
) -> Tuple[np.array, Tuple[int, int, int, int]]:
    """
    This function takes an image and a mask for this image (shape: (h, w))
    and returns a segmented chemical structure depiction (np.array)

    Args:
        image (np.array): image of a page from a scientific publication
        masks (np.array): binary mask (shape: (h, w))

    Returns:
        np.array: segmented chemical structure depiction
        Tuple[int]: (y0, x0, y1, x1)
    """
    # Get masked image and bbox more efficiently
    masked_image, bbox = get_masked_image_optimized(image, mask)
    x, y, w, h = bbox

    # Convert to grayscale more efficiently
    im_gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    _, im_bw = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Create alpha channel
    _, alpha = cv2.threshold(im_bw, 0, 255, cv2.THRESH_BINARY)

    # Extract region of interest first to reduce processing
    roi = image[y : y + h, x : x + w]

    # Split channels and merge with alpha
    b, g, r = cv2.split(roi)
    rgba = cv2.merge([b, g, r, alpha[y : y + h, x : x + w]])

    # Set transparent pixels to white
    trans_mask = rgba[:, :, 3] == 0
    rgba[trans_mask] = [255, 255, 255, 255]

    return rgba, (y, x, y + h, x + w)


def get_masked_image_optimized(
    image: np.array, mask: np.array
) -> Tuple[np.array, Tuple[int, int, int, int]]:
    """
    Optimized version of get_masked_image using vectorized operations
    """
    # Find bounding box more efficiently
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox = (cmin, rmin, cmax - cmin + 1, rmax - rmin + 1)

    # Create output image
    masked_image = np.zeros_like(image, dtype=np.uint8)
    # Apply mask using vectorized operation
    for c in range(3):
        masked_image[:, :, c] = mask * 255

    return masked_image, bbox


def save_images(images: List[np.array], path: str, name: str) -> None:
    """
    This function takes an array of np.array images, an output path
    and an ID for the name generation and saves the images as png files
    ("$name_$index.png).

    Args:
        images (List[np.array]): Images
        path (str): Output directory
        name (str): name for filename generation
    """
    os.makedirs(path, exist_ok=True)

    # Use thread pool for parallel I/O operations
    def save_single_image(args):
        index, image = args
        filename = f"{name}_{index}.png"
        file_path = os.path.join(path, filename)
        cv2.imwrite(file_path, image)

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(save_single_image, enumerate(images))


def get_bnw_image(image: np.array) -> np.array:
    """
    This function takes an image and returns a binarized version

    Args:
        image (np.array): input image

    Returns:
        np.array: binarized input image
    """
    # Check if already grayscale
    if len(image.shape) == 2:
        grayscale_im = image
    else:
        grayscale_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, im_bw = cv2.threshold(
        grayscale_im, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return im_bw


def get_square_image(image: np.array, desired_size: int) -> np.array:
    """
    This function takes an image and resizes it without distortion
    with the result of a square image with an edge length of
    desired_size.

    Args:
        image (np.array): input image
        desired_size (int): desired output image length/height

    Returns:
        np.array: resized output image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = image

    old_height, old_width = grayscale.shape

    # Calculate new size
    if old_height != desired_size or old_width != desired_size:
        ratio = float(desired_size) / max(old_height, old_width)
        new_width = int(old_width * ratio)
        new_height = int(old_height * ratio)

        # Resize using OpenCV (faster than PIL)
        resized = cv2.resize(
            grayscale, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
        )
    else:
        new_width, new_height = old_width, old_height
        resized = grayscale

    # Create output image
    output = np.full((desired_size, desired_size), 255, dtype=np.uint8)

    # Calculate padding
    y_offset = (desired_size - new_height) // 2
    x_offset = (desired_size - new_width) // 2

    # Place resized image in center
    output[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized

    return output


def get_model():
    """Get or create the global model instance"""
    global _model
    if _model is None:
        _model = load_model()
    return _model


def main():
    """
    This script takes a file path as an argument (pdf or image), runs DECIMER
    Segmentation on it and saves the segmented structures as PNG images.
    """
    # Handle input arguments
    description = "Segment chemical structures from the scientific literature"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input", help="Enter the input filename (pdf or image)", required=True
    )
    args = parser.parse_args()

    # Define image path and output path
    input_path = os.path.normpath(args.input)

    # Pre-load model before segmentation
    print("Loading model...")
    get_model()  # Pre-load the model

    # Segment chemical structure depictions
    print("Segmenting structures...")
    segments = segment_chemical_structures_from_file(input_path)

    # Save segments
    segment_dir = os.path.join(f"{input_path}_output", "segments")
    save_images(segments, segment_dir, os.path.split(input_path)[1][:-4])
    print(f"The segmented images can be found in {segment_dir}")


if __name__ == "__main__":
    main()
