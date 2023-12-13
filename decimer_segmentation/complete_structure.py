import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, binary_dilation
from typing import List, Tuple
from scipy.ndimage import label


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
    binarized_image_array = grayscale > threshold
    return binarized_image_array


def get_seeds(
    image_array: np.array,
    mask_array: np.array,
    exclusion_mask: np.array,
) -> List[Tuple[int, int]]:
    """
    This function takes an array that represents an image and a mask.
    It returns a list of tuples with indices of seeds in the structure
    covered by the mask.

    The seed pixels are defined as pixels in the inner 80% of the mask which
    are not white in the image.

    Args:
        image_array (np.array): Image
        mask_array (np.array): Mask array of shape (y, x)
        exclusion_mask (np.array): Exclusion mask

    Returns:
        List[Tuple[int, int]]: [(x,y), (x,y), ...]
    """
    mask_y_values, mask_x_values = np.where(mask_array)
    # Define boundaries of the inner 80% of the mask
    mask_y_diff = mask_y_values.max() - mask_y_values.min()
    mask_x_diff = mask_x_values.max() - mask_x_values.min()
    x_min_limit = mask_x_values.min() + mask_x_diff / 10
    x_max_limit = mask_x_values.max() - mask_x_diff / 10
    y_min_limit = mask_y_values.min() + mask_y_diff / 10
    y_max_limit = mask_y_values.max() - mask_y_diff / 10
    # Define intersection of mask and image
    mask_coordinates = set(zip(mask_y_values, mask_x_values))
    image_y_values, image_x_values = np.where(np.invert(image_array))
    image_coordinates = set(zip(image_y_values, image_x_values))

    intersection_coordinates = mask_coordinates & image_coordinates
    # Select intersection coordinates that are in the inner 80% of the mask
    seed_pixels = []
    for y_coord, x_coord in intersection_coordinates:
        if x_coord < x_min_limit:
            continue
        if x_coord > x_max_limit:
            continue
        if y_coord < y_min_limit:
            continue
        if y_coord > y_max_limit:
            continue
        if exclusion_mask[y_coord, x_coord]:
            continue
        seed_pixels.append((x_coord, y_coord))
    return seed_pixels


def detect_horizontal_and_vertical_lines(
    image: np.ndarray, max_depiction_size: Tuple[int, int]
) -> np.ndarray:
    """
    This function takes an image and returns a binary mask that labels the pixels that
    are part of long horizontal or vertical lines.

    Args:
        image (np.ndarray): binarised image (np.array; type bool) as it is returned by
            binary_erosion() in complete_structure_mask()
        max_depiction_size (Tuple[int, int]): height, width; used as thresholds

    Returns:
        np.ndarray: Exclusion mask that contains indices of pixels that are part of
            horizontal or vertical lines
    """
    binarised_im = ~image * 255
    binarised_im = binarised_im.astype("uint8")

    structure_height, structure_width = max_depiction_size

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (structure_width, 1))
    horizontal_mask = cv2.morphologyEx(
        binarised_im, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    horizontal_mask = horizontal_mask == 255

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, structure_height))
    vertical_mask = cv2.morphologyEx(
        binarised_im, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )
    vertical_mask = vertical_mask == 255

    return horizontal_mask + vertical_mask


def find_equidistant_points(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    num_points: int = 5
) -> List[Tuple[int, int]]:
    """
    Finds equidistant points between two points.

    Args:
        x1 (int): x coordinate of first point
        y1 (int): y coordinate of first point
        x2 (int): x coordinate of second point
        y2 (int): y coordinate of second point
        num_points (int, optional): Number of points to return. Defaults to 5.

    Returns:
        List[Tuple[int, int]]: Equidistant points on the given line
    """
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        x = x1 * (1 - t) + x2 * t
        y = y1 * (1 - t) + y2 * t
        points.append((x, y))
    return points


def detect_lines(
    image: np.ndarray,
    max_depiction_size: Tuple[int, int],
    segmentation_mask: np.ndarray
) -> np.ndarray:
    """
    This function takes an image and returns a binary mask that labels the pixels that
    are part of lines that are not part of chemical structures (like arrays, tables).

    Args:
        image (np.ndarray): binarised image (np.array; type bool) as it is returned by
            binary_erosion() in complete_structure_mask()
        max_depiction_size (Tuple[int, int]): height, width; used for thresholds
        segmentation_mask (np.ndarray): Indicates whether or not a pixel is part of a
            chemical structure depiction (shape: (height, width))
    Returns:
        np.ndarray: Exclusion mask that contains indices of pixels that are part of
            horizontal or vertical lines
    """
    image = ~image * 255
    image = image.astype("uint8")
    # Detect lines using the Hough Transform
    lines = cv2.HoughLinesP(image,
                            1,
                            np.pi / 180,
                            threshold=5,
                            minLineLength=int(max(max_depiction_size)/4),
                            maxLineGap=10)
    # Generate exclusion mask based on detected lines
    exclusion_mask = np.zeros_like(image)
    if lines is None:
        return exclusion_mask
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Check if any of the lines is in a chemical structure depiction
        points = find_equidistant_points(x1, y1, x2, y2, num_points=7)
        points_in_structure = False
        for x, y in points[1:-1]:
            if segmentation_mask[int(y), int(x)]:
                points_in_structure = True
                break
        if points_in_structure:
            continue
        cv2.line(exclusion_mask, (x1, y1), (x2, y2), 255, 2)
    return exclusion_mask


def expand_masks(
    image_array: np.array,
    seed_pixels: List[Tuple[int, int]],
    mask_array: np.array,
) -> np.array:
    """
    This function generates a mask array where the given masks have been
    expanded to surround the covered object in the image completely.

    Args:
        image_array (np.array): array that represents an image (float values)
        seed_pixels (List[Tuple[int, int]]): [(x, y), ...]
        mask_array (np.array): MRCNN output; shape: (y, x, mask_index)

    Returns:
        np.array: Expanded masks
    """
    image_array = np.invert(image_array)
    labeled_array, _ = label(image_array)
    mask_array = np.zeros_like(image_array)
    for seed_pixel in seed_pixels:
        x, y = seed_pixel
        if mask_array[y, x]:
            continue
        label_value = labeled_array[y, x]
        if label_value > 0:
            mask_array[labeled_array == label_value] = True
    return mask_array


def expansion_coordination(
    mask_array: np.array, image_array: np.array, exclusion_mask: np.array
) -> np.array:
    """
    This function takes a single mask and an image (np.array) and coordinates
    the mask expansion. It returns the expanded mask. The purpose of this function is
    wrapping up the expansion procedure in a map function.
    """
    seed_pixels = get_seeds(image_array,
                            mask_array,
                            exclusion_mask)
    mask_array = expand_masks(image_array, seed_pixels, mask_array)
    return mask_array


def complete_structure_mask(
    image_array: np.array,
    mask_array: np.array,
    max_depiction_size: Tuple[int, int],
    debug=False,
) -> np.array:
    """
    This funtion takes an image (np.array) and an array containing the masks (shape:
    x,y,n where n is the amount of masks and x and y are the pixel coordinates).
    Additionally, it takes the maximal depiction size of the structures in the image
    which is used to define the kernel size for the vertical and horizontal line
    detection for the exclusion masks. The exclusion mask is used to exclude pixels
    from the mask expansion to avoid including whole tables.
    It detects objects on the contours of the mask and expands it until it frames the
    complete object in the image. It returns the expanded mask array

    Args:
        image_array (np.array): input image
        mask_array (np.array): shape: y, x, n where n is the amount of masks
        max_depiction_size (Tuple[int, int]): height, width
        debug (bool, optional): You get visualisations in a Jupyter Notebook if True.
            Defaults to False.

    Returns:
        np.array: expanded mask array
    """

    if mask_array.size != 0:
        # Binarization of input image
        binarized_image_array = binarize_image(image_array, threshold=0.72)
        if debug:
            plot_it(binarized_image_array)
        # Apply dilation with a resolution-dependent kernel to the image
        blur_factor = (
            int(image_array.shape[1] / 185) if image_array.shape[1] / 185 >= 2 else 2
        )
        kernel = np.ones((blur_factor, blur_factor))
        blurred_image_array = binary_erosion(binarized_image_array, footprint=kernel)
        if debug:
            plot_it(blurred_image_array)
        # Slice mask array along third dimension into single masks
        split_mask_arrays = np.array(
            [mask_array[:, :, index] for index in range(mask_array.shape[2])]
        )
        # Detect horizontal and vertical lines
        horizontal_vertical_lines = detect_horizontal_and_vertical_lines(
            blurred_image_array, max_depiction_size
        )

        hough_lines = detect_lines(
            binarized_image_array,
            max_depiction_size,
            segmentation_mask=np.any(mask_array, axis=2).astype(np.bool)
        )
        hough_lines = binary_dilation(hough_lines, footprint=kernel)
        exclusion_mask = horizontal_vertical_lines + hough_lines
        image_with_exclusion = np.invert(
            np.invert(blurred_image_array) * np.invert(exclusion_mask)
        )
        if debug:
            plot_it(horizontal_vertical_lines)
            plot_it(hough_lines)
            plot_it(exclusion_mask)
            plot_it(image_with_exclusion)
        # Run expansion
        image_repeat = itertools.repeat(image_with_exclusion, mask_array.shape[2])
        exclusion_repeat = itertools.repeat(exclusion_mask, mask_array.shape[2])
        # Faster with map function
        expanded_split_mask_arrays = map(
            expansion_coordination,
            split_mask_arrays,
            image_repeat,
            exclusion_repeat,
        )
        # Filter duplicates and stack mask arrays to give the desired output format
        expanded_split_mask_arrays = filter_duplicate_masks(expanded_split_mask_arrays)
        mask_array = np.stack(expanded_split_mask_arrays, -1)
        return mask_array
    else:
        print("No masks found.")
        return mask_array


def filter_duplicate_masks(array_list: List[np.array]) -> List[np.array]:
    """
    This function takes a list of arrays and returns a list of unique arrays.

    Args:
        array_list (List[np.array]): Masks

    Returns:
        List[np.array]: Unique masks
    """
    seen = set()
    unique_list = []
    for arr in array_list:
        # Convert the array to a hashable tuple
        arr_tuple = tuple(arr.ravel())
        if arr_tuple not in seen:
            seen.add(arr_tuple)
            unique_list.append(arr)
    return unique_list
