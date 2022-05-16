import numpy as np
from decimer_segmentation import get_bounding_box_center


def test_get_bounding_box_center():
    # Determine the center of a given polygon bounding box
    test_bbox = np.array([[1, 1], [2, 1], [3, 1], [3, 0], [2, 0], [1, 0]])
    expected_result = np.array([2, 0.5])
    actual_result = get_bounding_box_center(test_bbox)
    for index in range(len(expected_result)):
        assert expected_result[index] == actual_result[index]


def test_get_edge_line():
    pass


def test_set_x_range():
    pass
