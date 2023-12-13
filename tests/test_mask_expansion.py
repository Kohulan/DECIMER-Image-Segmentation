import numpy as np
from decimer_segmentation.complete_structure import (
    binarize_image,
    get_seeds,
    expand_masks,
    expansion_coordination,
    detect_horizontal_and_vertical_lines,
)


def test_binarize_image():
    test_image_array = np.array([[255, 255, 255],[0, 0, 0],[255, 255, 255]])
    test_threshold = "otsu"
    expected_result = np.array([True, False, True])
    actual_result = binarize_image(test_image_array, test_threshold)
    assert np.array_equal(expected_result, actual_result)


def test_get_seeds():
    test_image_array = np.array([[0, 1, 0],[1, 0, 1],[0, 1, 0]])
    test_mask_array = np.ones(test_image_array.shape)
    exclusion_mask = np.zeros(test_image_array.shape)
    expected_result = []
    actual_result = get_seeds(test_image_array, test_mask_array, exclusion_mask)
    for index in range(len(expected_result)):
        assert expected_result[index] == actual_result[index]


def test_expand_masks():
    test_image_array = np.array([(False, False, True, True, True)])
    test_seed_pixels = [(2, 0)]
    test_mask_array = np.array([(True, True, False, True, True)])
    expected_result = np.array([(False, False, True, True, True)])
    actual_result = expand_masks(
        test_image_array,
        test_seed_pixels,
        test_mask_array,
    )
    assert expected_result.all() == actual_result.all()


def test_expansion_coordination():
    test_image_array = np.array([(False, False, True, True, True)])
    test_mask_array = np.array([(True, True, False, True, True)])
    expected_result = np.array([(False, False, True, True, True)])
    actual_result = expansion_coordination(
        test_mask_array, test_image_array, np.zeros(test_image_array.shape, dtype=bool)
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
