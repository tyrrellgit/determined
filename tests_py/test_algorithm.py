import unittest
import numpy as np

import determined as dt

from harness.models import (
    LinearMeasurement,
    LinearTransition,
    LinearUpdate
)

class TestAlgorithm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.value = np.array([1.0, 1.0])
        cls.cov = np.eye(cls.value.size)
        cls.epoch = dt.Epoch(0)
        cls.obs = dt.Observation(np.array([0.0, 1.0]), cls.epoch)

        cls.f = np.diag(np.array([2.0, 1.5]))
        cls.q = np.eye(cls.f.shape[0])
        cls.h = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        cls.r = np.eye(cls.h.shape[0])

        cls.one_step_state = cls.f @ cls.value
        return super().setUpClass()
    
    # Rebuild models at each test to avoid persistent state
    def setUp(self):
        self.state = dt.State(self.value, self.cov, self.epoch)

        self.t = LinearTransition(self.f, self.q, self.state)
        self.transition = dt.TransitionModel(self.t, self.state)

        self.m = LinearMeasurement(self.h, self.r)
        self.measurement = dt.MeasurementModel(self.m)

        self.u = LinearUpdate(self.transition, self.measurement)
        self.update = dt.UpdateModel(self.u, self.transition, self.measurement)

        # this is basically just a type-erased wrapper
        self.algorithm = dt.KalmanFilter(self.update)
        return super().setUp()
    
    def test_predict(self):
        _e = dt.Epoch(1)
        _state = self.algorithm.predict(_e)

        # one step update should ALWAYS be the result --> according to harness model
        np.testing.assert_array_equal(_state.value, self.one_step_state)

        # check the original state object has also been updated
        np.testing.assert_array_equal(_state.value, self.state.value)
        np.testing.assert_array_equal(_state.covariance, self.state.covariance)
        np.testing.assert_equal(_state.epoch.value, self.state.epoch.value)

    def test_apply(self):
        new_state = self.algorithm.update(self.obs)
        assert isinstance(new_state, dt.State), "state should be dt.State"
