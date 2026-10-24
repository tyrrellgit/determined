import unittest
import numpy as np

import determined as dt

from harness.models import LinearMeasurement


class TestMeasurementModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.value = np.array([1.0, 0.0])
        cls.epoch = dt.Epoch(0)
        cls.state = dt.State(cls.value, np.eye(cls.value.size), cls.epoch)

        h = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        r = np.eye(h.shape[0])

        cls.model = LinearMeasurement(h, r)

        return super().setUpClass()
    
    def setUp(self):
        self.measurement = dt.MeasurementModel(self.model)
        return super().setUp()

    def test_projection(self):
        obs = self.measurement.projection(self.state)
        inv = self.measurement.inverse(obs)
        
        np.testing.assert_array_equal(inv.value, self.state.value)

    def test_jacobian(self):
        jac = self.measurement.jacobian(self.state)

        assert jac is not None, "Jacobian is None!"
