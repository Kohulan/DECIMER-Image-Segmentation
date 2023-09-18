import numpy as np
from decimer_segmentation.decimer_segmentation import *

def test_determine_depiction_size_with_buffer():
    # Determine the average depiction size of the structures in a given list of structures
    bboxes = [
        [0, 0, 6, 6],
        [0, 0, 8, 8],
        [0, 0, 10, 10],
    ]
    expected_result = (11, 11)
    actual_result = determine_depiction_size_with_buffer(bboxes)
    assert expected_result == actual_result
