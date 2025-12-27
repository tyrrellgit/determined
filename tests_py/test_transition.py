import unittest
import numpy as np

import determined as dt

from harness.models import LinearTransition


class TestTransisionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.value = np.array([1.0, 1.0])
        cls.cov = np.eye(cls.value.size)

        cls.epoch = dt.Epoch(0)
        cls.state = dt.State(cls.value, cls.cov, cls.epoch)

        f = np.diag(np.array([2.0, 1.5]))
        q = np.eye(f.shape[0])

        cls.one_step_state = f @ cls.value

        cls.model = LinearTransition(f, q, cls.state)
        return super().setUpClass()
    
    def setUp(self):
        self.transition = dt.TransitionModel(self.model, self.state)
        return super().setUp()
    
    def test_transition(self):
        _e = dt.Epoch(1)
        _state = self.transition.state(_e)
        np.testing.assert_array_equal(_state.value, self.one_step_state)

    def test_jacobian(self):
        jac = self.transition.jacobian(self.state)
        assert jac is not None, "Jacobian is None!"
