import numpy as np
from decimer_segmentation.complete_structure import *


def test_get_bounding_box_center():
    # Determine the center of a given polygon bounding box
    test_bbox = np.array([[1, 1], [2, 1], [3, 1], [3, 0], [2, 0], [1, 0]])
    expected_result = np.array([2, 0.5])
    actual_result = get_bounding_box_center(test_bbox)
    for index in range(len(expected_result)):
        assert expected_result[index] == actual_result[index]


def test_get_edge_line():
    # Return intercept and slop for a linear function between 2 points in 2D Space
    test_linenode1 = [1, 6]
    test_linenode2 = [5, 8]
    expected_result = [0.5, 5.5]
    actual_result = get_edge_line(test_linenode1, test_linenode2)
    for index in range(len(expected_result)):
        assert expected_result[index] == actual_result[index]


def test_get_euklidian_distance():
    # Calculates euklidian distance between two given points in 2D Space
    test_distancepoint1 = [1, 6]
    test_distancepoint2 = [5, 8]
    expected_result = 4.47213595499958
    actual_result = get_euklidian_distance(test_distancepoint1, test_distancepoint2)
    assert expected_result == actual_result


def test_set_x_range():
    # For the contour-based expansion, non-white pixels on the contours of the original polygon bounding box are detected
    test_distance = 3
    test_eukl_distance = 4
    test_image_array = np.array([[1, 5]])
    expected_result = [2.5, 2.75, 1.0, 0.25, 0.75, 0.0, 2.0, 2.25, 1.5, 1.75, 1.25, 0.5]
    actual_result = set_x_range(test_distance, test_eukl_distance, test_image_array)
    assert set(expected_result) == set(actual_result)


def test_get_next_pixel_to_check():
    # Returns the next pixel to check in the image
    test_bounding_box = np.array([[1, 5], [2, 4]])
    test_node_index = 1
    test_step = 4
    test_image_shape = [2, 4, 6]
    expected_result = (3, 1)
    actual_result = get_next_pixel_to_check(
        test_bounding_box, test_node_index, test_step, test_image_shape
    )
    for index in range(len(expected_result)):
        assert expected_result[index] == actual_result[index]


def test_adapt_x_values():
    # Returns a bounding box where the nodes are altered depending on their relative position to bounding box centre
    test_bounding_box = np.array([[1, 5], [2, 4]])
    test_node_index = 1
    test_image_shape = [2, 4, 6]
    expected_result = np.array([[1, 5], [3, 4]])
    actual_result = adapt_x_values(test_bounding_box, test_node_index, test_image_shape)
    assert expected_result.all() == actual_result.all()


def test_binarize_image():
    # Returns the binarized image (np.array) by applying the otsu threshold
    # test_image_array = np.array([1,2,3])
    # test_threshold = "otsu"
    # expected_result = False
    # actual_result = binarize_image(test_image_array, test_threshold)
    # assert expected_result == actual_result
    pass


def test_get_biggest_polygon():
    # returns the Polygon object that only contains the biggest bounding box
    # test_polygon = np.array([[(7,7), (8,16)]])
    # expected_result = np.array([[(7,7), (8,16)]])
    # actual_result = get_biggest_polygon(test_polygon)
    # assert expected_result.all() == actual_result.all()
    pass


def test_get_contour_seeds():
    test_image_array = np.array([(1, 2)])
    test_bounding_box = np.array([[1, 1], [2, 1]])
    expected_result = []
    actual_result = get_contour_seeds(test_image_array, test_bounding_box)
    for index in range(len(expected_result)):
        assert expected_result[index] == actual_result[index]


def test_get_mask_center():
    test_mask_array = np.array([(9, 5, 9, 5)])
    expected_result = (1, 0)
    actual_result = get_mask_center(test_mask_array)
    for index in range(len(expected_result)):
        assert expected_result[index] == actual_result[index]


def test_get_seeds():
    test_image_array = np.array([(3, 2)])
    test_mask_array = np.array([(9, 5, 9)])
    expected_result = []
    actual_result = get_seeds(test_image_array, test_mask_array)
    for index in range(len(expected_result)):
        assert expected_result[index] == actual_result[index]


def test_get_neighbour_pixels():
    test_seed_pixel = [2, 4]
    test_image_shape = [9, 2, 4]
    expected_result = [(1, 3), (1, 4), (1, 5)]
    actual_result = get_neighbour_pixels(test_seed_pixel, test_image_shape)
    for index in range(len(expected_result)):
        assert expected_result[index] == actual_result[index]


def test_expand_masks():
    test_image_array = np.array([(3, 2)])
    test_seed_pixels = [(2, 4)]
    test_mask_array = np.array([(9, 5, 9)])
    expected_result = np.array([(0.0, 0.0, 0.0)])
    actual_result = expand_masks(test_image_array, test_seed_pixels, test_mask_array)
    assert expected_result.all() == actual_result.all()


def test_expansion_coordination():
    test_mask_array = np.array([(9, 5, 9)])
    test_image_array = np.array([(3, 2)])
    expected_result = np.array([(9, 5, 9)])
    actual_result = expansion_coordination(test_mask_array, test_image_array)
    assert expected_result.all() == actual_result.all()


def test_complete_structure_mask():
    pass
