import numpy as np
from decimer_segmentation.decimer_segmentation import *

def test_determine_average_depiction_size():
    # Determine the average depiction size of the structures in a given list of structures
    bboxes = [
        [0, 0, 6, 6],
        [0, 0, 8, 8],
        [0, 0, 10, 10],
    ]
    expected_result = (8, 8)
    actual_result = determine_average_depiction_size(bboxes)
    assert expected_result == actual_result
