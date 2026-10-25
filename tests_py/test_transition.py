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
        cls.state = dt.State(cls.value, covariance=cls.cov, epoch=cls.epoch)

        f = np.diag(np.array([2.0, 1.5]))
        q = np.eye(f.shape[0])

        cls.one_step_state = f @ cls.value

        cls.model = LinearTransition(f, q, cls.state)
        return super().setUpClass()
    
    def setUp(self):
        self.transition = dt.TransitionModel(self.model, state=self.state)
        return super().setUp()
    
    def test_transition(self):
        _e = dt.Epoch(1)
        _state = self.transition.state(_e)
        np.testing.assert_array_equal(_state.value, self.one_step_state)

    def test_jacobian(self):
        jac = self.transition.jacobian(self.state)
        assert jac is not None, "Jacobian is None!"


class TestSharedState(unittest.TestCase):
    """Python models may return the State object they were built with."""

    def test_transition_returning_its_own_state(self):
        class Static:
            def __init__(self, state): self._state = state
            def state(self, epoch): return self._state
            def jacobian(self, state): return np.eye(2)

        state = dt.State(np.array([1.0, 2.0]), covariance=np.eye(2), epoch=dt.Epoch(0))
        transition = dt.TransitionModel(Static(state), state=state)
        out = transition.state(dt.Epoch(1))
        np.testing.assert_array_equal(out.value, [1.0, 2.0])
        self.assertEqual(out.epoch.value, 1)

    def test_update_returning_its_own_state(self):
        class Static:
            def __init__(self, state): self._state = state
            def state(self, epoch): return self._state
            def jacobian(self, state): return np.eye(2)

        class Measurement:
            def projection(self, s): return dt.Observation(s.value, epoch=s.epoch)
            def noise(self): return np.eye(2)

        class Update:
            def __init__(self, state): self._state = state
            def apply(self, observation): return self._state

        state = dt.State(np.array([1.0, 2.0]), covariance=np.eye(2), epoch=dt.Epoch(0))
        transition = dt.TransitionModel(Static(state), state=state)
        update = dt.UpdateModel(Update(state), transition=transition, measurement=dt.MeasurementModel(Measurement()))
        out = update.apply(dt.Observation(np.array([0.0, 0.0]), epoch=dt.Epoch(0)))
        np.testing.assert_array_equal(out.value, [1.0, 2.0])
