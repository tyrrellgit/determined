import unittest
import numpy as np

import determined as dt


class TestState(unittest.TestCase):
    
    def test_constructor(self):
        value = np.array([1.0, 0.0])
        cov = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        epoch = dt.Epoch(0)
        state = dt.State(value, cov, epoch)
        
        np.testing.assert_array_equal(value, state.value, "state value arrays not equal")
