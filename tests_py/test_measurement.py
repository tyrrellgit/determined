import unittest
import numpy as np

import determined as dt

from harness.models import LinearMeasurement


class TestMeasurementModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.value = np.array([1.0, 0.0])
        cls.epoch = dt.Epoch(0)
        cls.state = dt.State(cls.value, covariance=np.eye(cls.value.size), epoch=cls.epoch)

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

    def test_noise(self):
        np.testing.assert_array_equal(self.measurement.noise(), self.model.r)

    def test_inverse_may_be_none(self):
        class NoInverse:
            def projection(self, state): return dt.Observation(state.value, epoch=state.epoch)
            def inverse(self, observation): return None
            def noise(self): return np.eye(2)

        measurement = dt.MeasurementModel(NoInverse())
        obs = measurement.projection(self.state)
        self.assertIsNone(measurement.inverse(obs))
