import numpy as np
import pytest

from base_test import BaseTest

class TestHelperFunctions(BaseTest):

    def test_box_points(self):
        from morphct.helper_functions import box_points

        box = np.array([10,10,10])
        assert np.array_equal(
                box_points(box),
                np.array([[-0.5, -0.5, -0.5],
                          [-0.5, -0.5,  0.5],
                          [-0.5,  0.5,  0.5],
                          [-0.5, -0.5,  0.5],
                          [ 0.5, -0.5,  0.5],
                          [ 0.5, -0.5, -0.5],
                          [-0.5, -0.5, -0.5],
                          [-0.5,  0.5, -0.5],
                          [-0.5,  0.5,  0.5],
                          [ 0.5,  0.5,  0.5],
                          [ 0.5, -0.5,  0.5],
                          [ 0.5, -0.5, -0.5],
                          [ 0.5,  0.5, -0.5],
                          [-0.5,  0.5, -0.5],
                          [ 0.5,  0.5, -0.5],
                          [ 0.5,  0.5,  0.5]])
                )

    def test_time_units(self):
        from morphct.helper_functions import time_units

        assert time_units(10) == '10.00 seconds'
        assert time_units(1000, precision) == '17 minutes'
        assert time_units(10000) == '2.78 hours'
        assert time_units(100000) == '1.16 days'

    def test_find_axis(self):
        from morphct.helper_functions import find_axis

        assert np.array_equal(
                find_axis(np.array([0,0,0]), np.array([1,0,0])),
                np.array([1,0,0])
                )
        assert np.allclose(
                find_axis(np.array([0,0,0]), np.array([1,1,1])),
                np.array([0.57735027, 0.57735027, 0.57735027])
                )
        assert np.array_equal(
                find_axis(
                    np.array([0,0,0]), np.array([1,1,1]), normalize=False
                    ),
                np.array([1,1,1])
                )
        assert np.array_equal(
                find_axis(np.array([0,0,0]), np.array([0,0,0,])),
                np.array([0,0,0])
                )

    def test_
        from morphct.helper_functions import time_units
