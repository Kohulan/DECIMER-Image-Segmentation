import numpy as np
from decimer_segmentation.complete_structure import (
    get_seeds,
    expand_masks,
    expansion_coordination,
    detect_horizontal_and_vertical_lines,
)



def test_binarize_image():
    # Returns the binarized image (np.array) by applying the otsu threshold
    # test_image_array = np.array([1,2,3])
    # test_threshold = "otsu"
    # expected_result = False
    # actual_result = binarize_image(test_image_array, test_threshold)
    # assert expected_result == actual_result
    pass


def test_get_seeds():
    test_image_array = np.array([[0, 1, 0],[1, 0, 1],[0, 1, 0]])
    test_mask_array = np.ones(test_image_array.shape)
    exclusion_mask = np.zeros(test_image_array.shape)
    expected_result = []
    actual_result = get_seeds(test_image_array, test_mask_array, exclusion_mask)
    for index in range(len(expected_result)):
        assert expected_result[index] == actual_result[index]


def test_expand_masks():
    # TODO: Fix this mess! This is not testing anything.
    test_image_array = np.array([(0, 0, 0, 255, 0)])
    test_seed_pixels = [(2, 0)]
    test_mask_array = np.array([(True, True, True, True, True)])
    expected_result = np.array([(True, True, True, False, True)])
    actual_result = expand_masks(
        test_image_array,
        test_seed_pixels,
        test_mask_array,
        np.zeros(test_image_array.shape),
    )
    expected_result.all() == actual_result.all()
    # assert expected_result.all() == actual_result.all()


def test_expansion_coordination():
    # TODO: Go through tests and fix nonsense like this
    test_mask_array = np.array([(9, 5, 9)])
    test_image_array = np.array([(3, 2)])
    expected_result = np.array([(9, 5, 9)])
    actual_result = expansion_coordination(
        test_mask_array, test_image_array, np.zeros(test_image_array.shape)
    )
    assert expected_result.all() == actual_result.all()


def test_detect_horizontal_and_vertical_lines():
    test_image = np.array([[False] * 20] * 20)
    test_image[9] = np.array([True] * 20)
    test_image[2][3] = True
    expected_result = np.array([[False] * 20] * 20)
    expected_result[9] = np.array([True] * 20)
    actual_result = detect_horizontal_and_vertical_lines(~test_image, (2, 2))
    np.testing.assert_array_equal(expected_result, actual_result)


def test_complete_structure_mask():
    pass
