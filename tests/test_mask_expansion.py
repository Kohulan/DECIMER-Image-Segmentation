import sys
import os
import numpy as np
test_dir = os.path.split('__file__')[0]
main_path = os.path.join(test_dir, '..')
sys.path.append(main_path)
from Scripts.complete_structure import *

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
        

