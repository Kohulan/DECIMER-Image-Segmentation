import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, binary_dilation
from typing import List, Tuple
from scipy.ndimage import label
from numba import jit, prange
import warnings

# Suppress numba warnings for cleaner output
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def plot_it(image_array: np.array) -> None:
    """
    This function shows the plot of a given image (np.array)

    Args:
        image_array (np.array): Image
    """
    plt.rcParams["figure.figsize"] = (20, 15)
    _, ax = plt.subplots(1)
    ax.imshow(image_array)
    plt.show()


def binarize_image(image_array: np.array, threshold="otsu") -> np.array:
    """
    This function takes a np.array that represents an RGB image and returns
    the binarized image (np.array) by applying the otsu threshold.

    Args:
        image_array (np.array): image
        threshold (str, optional): "otsu" or a float. Defaults to "otsu".

    Returns:
        np.array: binarized image
    """
    grayscale = rgb2gray(image_array)
    if threshold == "otsu":
        threshold = threshold_otsu(grayscale)
    return grayscale > threshold


@jit(nopython=True, cache=True)
def _get_seeds_fast(
    mask_indices,
    image_indices,
    exclusion_indices,
    x_min_limit,
    x_max_limit,
    y_min_limit,
    y_max_limit,
):
    """
    Fast numba-compiled function for seed pixel detection.
    """
    # Convert to sets for fast intersection
    mask_set = set()
    for i in range(len(mask_indices[0])):
        mask_set.add((mask_indices[0][i], mask_indices[1][i]))

    image_set = set()
    for i in range(len(image_indices[0])):
        image_set.add((image_indices[0][i], image_indices[1][i]))

    exclusion_set = set()
    for i in range(len(exclusion_indices[0])):
        exclusion_set.add((exclusion_indices[0][i], exclusion_indices[1][i]))

    # Find intersection and filter
    seed_pixels = []
    for coord in mask_set:
        if coord in image_set:
            y_coord, x_coord = coord
            if (
                x_coord >= x_min_limit
                and x_coord <= x_max_limit
                and y_coord >= y_min_limit
                and y_coord <= y_max_limit
                and coord not in exclusion_set
            ):
                seed_pixels.append((x_coord, y_coord))

    return seed_pixels


def get_seeds(
    image_array: np.array,
    mask_array: np.array,
    exclusion_mask: np.array,
) -> List[Tuple[int, int]]:
    """
    Optimized version of get_seeds function.
    """
    mask_indices = np.where(mask_array)
    if len(mask_indices[0]) == 0:
        return []

    # Calculate boundaries once
    mask_y_diff = mask_indices[0].max() - mask_indices[0].min()
    mask_x_diff = mask_indices[1].max() - mask_indices[1].min()
    x_min_limit = mask_indices[1].min() + mask_x_diff / 10
    x_max_limit = mask_indices[1].max() - mask_x_diff / 10
    y_min_limit = mask_indices[0].min() + mask_y_diff / 10
    y_max_limit = mask_indices[0].max() - mask_y_diff / 10

    image_indices = np.where(~image_array)
    exclusion_indices = np.where(exclusion_mask)

    return _get_seeds_fast(
        mask_indices,
        image_indices,
        exclusion_indices,
        x_min_limit,
        x_max_limit,
        y_min_limit,
        y_max_limit,
    )


def detect_horizontal_and_vertical_lines(
    image: np.ndarray, max_depiction_size: Tuple[int, int]
) -> np.ndarray:
    """
    Optimized version with pre-allocated arrays and combined operations.
    """
    # Convert to uint8 once
    binarised_im = (~image).astype(np.uint8) * 255
    structure_height, structure_width = max_depiction_size

    # Use optimized kernel creation and operations
    horizontal_kernel = np.ones((1, structure_width), dtype=np.uint8)
    vertical_kernel = np.ones((structure_height, 1), dtype=np.uint8)

    # Perform morphological operations
    horizontal_mask = (
        cv2.morphologyEx(binarised_im, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        == 255
    )

    vertical_mask = (
        cv2.morphologyEx(binarised_im, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        == 255
    )

    return horizontal_mask | vertical_mask  # Use bitwise OR instead of addition


@jit(nopython=True, cache=True)
def _find_equidistant_points_fast(x1, y1, x2, y2, num_points):
    """
    Vectorized version of equidistant points calculation.
    """
    points = np.empty((num_points + 1, 2), dtype=np.float64)
    for i in range(num_points + 1):
        t = i / num_points
        points[i, 0] = x1 * (1 - t) + x2 * t
        points[i, 1] = y1 * (1 - t) + y2 * t
    return points


def find_equidistant_points(
    x1: int, y1: int, x2: int, y2: int, num_points: int = 5
) -> np.ndarray:
    """
    Optimized version using numba compilation.
    """
    return _find_equidistant_points_fast(x1, y1, x2, y2, num_points)


def detect_lines(
    image: np.ndarray,
    max_depiction_size: Tuple[int, int],
    segmentation_mask: np.ndarray,
) -> np.ndarray:
    """
    Optimized line detection with vectorized operations.
    """
    # Convert to uint8 once
    image_uint8 = (~image).astype(np.uint8) * 255

    # Detect lines using the Hough Transform
    lines = cv2.HoughLinesP(
        image_uint8,
        1,
        np.pi / 180,
        threshold=5,
        minLineLength=int(max(max_depiction_size) / 4),
        maxLineGap=10,
    )

    if lines is None:
        return np.zeros_like(image_uint8, dtype=np.uint8)

    # Pre-allocate exclusion mask
    exclusion_mask = np.zeros_like(image_uint8, dtype=np.uint8)

    # Vectorized line processing
    for line in lines:
        x1, y1, x2, y2 = line[0]
        points = find_equidistant_points(x1, y1, x2, y2, num_points=7)

        # Check if points are in structure (vectorized)
        valid_points = points[1:-1]  # Exclude endpoints
        coords = np.clip(
            valid_points.astype(int),
            0,
            [segmentation_mask.shape[1] - 1, segmentation_mask.shape[0] - 1],
        )

        if not segmentation_mask[coords[:, 1], coords[:, 0]].any():
            cv2.line(exclusion_mask, (x1, y1), (x2, y2), 255, 2)

    return exclusion_mask


def expand_masks(
    image_array: np.array,
    seed_pixels: List[Tuple[int, int]],
    mask_array: np.array,
) -> np.array:
    """
    Optimized mask expansion with reduced memory allocations.
    """
    if not seed_pixels:
        return np.zeros_like(image_array, dtype=bool)

    inverted_image = ~image_array
    labeled_array, _ = label(inverted_image)
    result_mask = np.zeros_like(image_array, dtype=bool)

    # Process all seed pixels in vectorized manner where possible
    processed_labels = set()
    for x, y in seed_pixels:
        if result_mask[y, x]:
            continue
        label_value = labeled_array[y, x]
        if label_value > 0 and label_value not in processed_labels:
            result_mask[labeled_array == label_value] = True
            processed_labels.add(label_value)

    return result_mask


def expansion_coordination(
    mask_array: np.array, image_array: np.array, exclusion_mask: np.array
) -> np.array:
    """
    Optimized coordination function.
    """
    seed_pixels = get_seeds(image_array, mask_array, exclusion_mask)
    return expand_masks(image_array, seed_pixels, mask_array)


def complete_structure_mask(
    image_array: np.array,
    mask_array: np.array,
    max_depiction_size: Tuple[int, int],
    debug=False,
) -> np.array:
    """
    Heavily optimized version of complete_structure_mask.
    """
    if mask_array.size == 0:
        print("No masks found.")
        return mask_array

    # Optimize binarization
    binarized_image_array = binarize_image(image_array, threshold=0.72)
    if debug:
        plot_it(binarized_image_array)

    # Calculate blur factor and kernel once
    blur_factor = max(2, int(image_array.shape[1] / 185))
    kernel = np.ones((blur_factor, blur_factor), dtype=np.uint8)

    # Apply erosion
    blurred_image_array = binary_erosion(binarized_image_array, footprint=kernel)
    if debug:
        plot_it(blurred_image_array)

    # Optimized mask splitting - use moveaxis instead of list comprehension
    split_mask_arrays = np.moveaxis(mask_array, 2, 0)

    # Detect lines with optimized functions
    horizontal_vertical_lines = detect_horizontal_and_vertical_lines(
        blurred_image_array, max_depiction_size
    )

    # Create segmentation mask more efficiently
    segmentation_mask = mask_array.any(axis=2)

    hough_lines = detect_lines(
        binarized_image_array,
        max_depiction_size,
        segmentation_mask=segmentation_mask,
    )

    # Combine masks efficiently
    hough_lines_dilated = binary_dilation(hough_lines, footprint=kernel)
    exclusion_mask = horizontal_vertical_lines | hough_lines_dilated

    # Optimize image processing
    image_with_exclusion = ~((~blurred_image_array) & (~exclusion_mask))

    if debug:
        plot_it(horizontal_vertical_lines)
        plot_it(hough_lines_dilated)
        plot_it(exclusion_mask)
        plot_it(image_with_exclusion)

    # Optimized expansion using list comprehension instead of map
    expanded_masks = [
        expansion_coordination(mask, image_with_exclusion, exclusion_mask)
        for mask in split_mask_arrays
    ]

    # Optimized duplicate filtering
    unique_masks = filter_duplicate_masks_fast(expanded_masks)

    if not unique_masks:
        return np.empty((image_array.shape[0], image_array.shape[1], 0), dtype=bool)

    return np.stack(unique_masks, axis=-1)


def filter_duplicate_masks_fast(array_list: List[np.array]) -> List[np.array]:
    """
    Highly optimized duplicate filtering using hash-based comparison.
    """
    if not array_list:
        return []

    seen_hashes = set()
    unique_list = []

    for arr in array_list:
        # Use hash of array for faster comparison
        if arr.size > 0:
            # Use a more efficient hash method
            arr_hash = hash(arr.tobytes())
            if arr_hash not in seen_hashes:
                seen_hashes.add(arr_hash)
                unique_list.append(arr)
        elif not seen_hashes:  # Handle empty arrays
            seen_hashes.add(0)  # Placeholder for empty array
            unique_list.append(arr)

    return unique_list


# Alternative filter method for very large arrays
def filter_duplicate_masks_memory_efficient(
    array_list: List[np.array],
) -> List[np.array]:
    """
    Memory-efficient version for very large arrays.
    """
    if not array_list:
        return []

    unique_list = []

    for i, arr1 in enumerate(array_list):
        is_duplicate = False
        for j in range(i):
            if np.array_equal(arr1, array_list[j]):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_list.append(arr1)

    return unique_list
