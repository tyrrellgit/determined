import unittest
import numpy as np

import determined as dt

from harness.models import (
    LinearMeasurement,
    LinearTransition,
    LinearUpdate
)


def make_ekf(state, f_mat, q, h_mat, r):
    return dt.ExtendedKalmanFilter(
        state,
        f=lambda x: f_mat @ x,
        jac_f=lambda x: f_mat,
        q=q,
        h=lambda x: h_mat @ x,
        jac_h=lambda x: h_mat,
        r=r,
    )


class TestExtendedKalmanFilter(unittest.TestCase):

    def test_predict(self):
        state = dt.State(np.array([1.0, 1.0]), np.eye(2), dt.Epoch(0))
        f = np.diag([2.0, 3.0])
        ekf = make_ekf(state, f, np.zeros((2, 2)), np.eye(2), np.eye(2))

        out = ekf.predict(dt.Epoch(2))
        np.testing.assert_allclose(out.value, [4.0, 9.0])
        np.testing.assert_allclose(out.covariance, np.diag([16.0, 81.0]))
        self.assertEqual(out.epoch.value, 2)

        # shares storage with the state it was built from
        np.testing.assert_array_equal(state.value, out.value)

        # predicting to the same epoch again must not propagate
        out = ekf.predict(dt.Epoch(2))
        np.testing.assert_allclose(out.value, [4.0, 9.0])

    def test_matches_harness_kalman_filter(self):
        """A linear problem must match the Python-defined linear KalmanFilter."""
        f = np.array([[1.0, 0.5], [0.0, 1.0]])   # deliberately not symmetric
        q = np.eye(2) * 0.05
        h = np.array([[1.0, 0.0]])                # non-square (1 x 2)
        r = np.array([[0.1]])
        value, cov = np.array([0.0, 1.0]), np.eye(2)

        ekf = make_ekf(dt.State(value, cov, dt.Epoch(0)), f, q, h, r)

        # reference: closed-form KF implemented independently in numpy
        x, p = value.copy(), cov.copy()
        for t in range(1, 5):
            z = np.array([0.4 * t])
            ekf.predict(dt.Epoch(t))
            out = ekf.update(dt.Observation(z, dt.Epoch(t)))

            x, p = f @ x, f @ p @ f.T + q
            s = h @ p @ h.T + r
            k = p @ h.T @ np.linalg.inv(s)
            x, p = x + k @ (z - h @ x), (np.eye(2) - k @ h) @ p

            np.testing.assert_allclose(out.value, x, atol=1e-12)
            np.testing.assert_allclose(out.covariance, p, atol=1e-12)

    def test_range_sensor(self):
        """Nonlinear range sensor pulls the estimate to the truth."""
        truth_range = 5.0
        state = dt.State(np.array([2.0, 2.0]), np.eye(2) * 4.0, dt.Epoch(0))
        ekf = dt.ExtendedKalmanFilter(
            state,
            f=lambda x: x,
            jac_f=lambda x: np.eye(2),
            q=np.zeros((2, 2)),
            h=lambda x: np.array([np.linalg.norm(x)]),
            jac_h=lambda x: (x / np.linalg.norm(x))[None, :],
            r=np.array([[0.01]]),
        )

        err = abs(np.linalg.norm(state.value) - truth_range)
        trace = np.trace(state.covariance)
        for t in range(1, 6):
            out = ekf.update(dt.Observation(np.array([truth_range]), dt.Epoch(t)))
            new_err = abs(np.linalg.norm(out.value) - truth_range)
            new_trace = np.trace(out.covariance)
            self.assertLess(new_err, err)
            self.assertLess(new_trace, trace)
            err, trace = new_err, new_trace
        self.assertLess(err, 1e-2)

    def test_callables_may_return_array_likes(self):
        """Lists / (N, 1) arrays are coerced, as AD libraries return their own types."""
        state = dt.State(np.array([1.0, 2.0]), np.eye(2), dt.Epoch(0))
        ekf = dt.ExtendedKalmanFilter(
            state,
            f=lambda x: [float(x[0]), float(x[1])],
            jac_f=lambda x: [[1, 0], [0, 1]],
            q=np.zeros((2, 2)),
            h=lambda x: x.reshape(2, 1),
            jac_h=lambda x: np.eye(2),
            r=np.eye(2),
        )
        out = ekf.update(dt.Observation(np.array([1.0, 2.0]), dt.Epoch(1)))
        np.testing.assert_allclose(out.value, [1.0, 2.0])

    def test_reset_and_state_setter(self):
        state = dt.State(np.array([1.0, 1.0]), np.eye(2) * 3.0, dt.Epoch(0))
        ekf = make_ekf(state, np.eye(2), np.eye(2), np.eye(2), np.eye(2))

        ekf.reset()
        np.testing.assert_array_equal(ekf.state.value, [0.0, 0.0])
        np.testing.assert_array_equal(ekf.state.covariance, np.eye(2))

        ekf.state = dt.State(np.array([5.0, 6.0]), np.eye(2), dt.Epoch(3))
        np.testing.assert_array_equal(ekf.state.value, [5.0, 6.0])
        self.assertEqual(ekf.state.epoch.value, 3)

    def test_bad_inputs_raise(self):
        state = dt.State(np.array([1.0, 1.0]), np.eye(2), dt.Epoch(0))
        eye = np.eye(2)

        with self.assertRaises(TypeError):
            dt.ExtendedKalmanFilter(state, 1, lambda x: eye, eye, lambda x: x, lambda x: eye, eye)
        with self.assertRaises(ValueError):
            dt.ExtendedKalmanFilter(state, lambda x: x, lambda x: eye, np.eye(3),
                                    lambda x: x, lambda x: eye, eye)

        # Jacobian of the wrong shape surfaces as ValueError, not a panic
        ekf = dt.ExtendedKalmanFilter(
            state, lambda x: x, lambda x: np.eye(3), eye,
            lambda x: x, lambda x: eye, eye)
        with self.assertRaises(ValueError):
            ekf.predict(dt.Epoch(1))

        # wrong observation size
        ekf = make_ekf(state, eye, eye, eye, eye)
        with self.assertRaises(ValueError):
            ekf.update(dt.Observation(np.array([1.0]), dt.Epoch(1)))

        # errors raised inside the callable propagate
        def boom(x):
            raise RuntimeError("boom")
        ekf = dt.ExtendedKalmanFilter(state, boom, lambda x: eye, eye,
                                      lambda x: x, lambda x: eye, eye)
        with self.assertRaises(RuntimeError):
            ekf.predict(dt.Epoch(1))


class TestMatrixLayout(unittest.TestCase):
    """Round trips of non-square / non-symmetric matrices through the bindings.

    These pass regardless of the internal layout; the non-square `H` in
    `test_matches_harness_kalman_filter` is what checks the layout is correct.
    """

    def test_nonsquare_jacobian_roundtrip(self):
        jac = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        class Measurement:
            def projection(self, s): return dt.Observation(np.zeros(2), s.epoch)
            def inverse(self, o): return dt.State(np.zeros(3), np.eye(3), o.epoch)
            def jacobian(self, s): return jac

        model = dt.MeasurementModel(Measurement())
        state = dt.State(np.zeros(3), np.eye(3), dt.Epoch(0))
        np.testing.assert_array_equal(model.jacobian(state), jac)

    def test_covariance_roundtrip(self):
        cov = np.array([[1.0, 2.0], [3.0, 4.0]])
        state = dt.State(np.zeros(2), cov, dt.Epoch(0))
        np.testing.assert_array_equal(state.covariance, cov)



class TestLogging(unittest.TestCase):
    """Rust-side log messages arrive through Python's `logging`."""

    def test_singular_update_is_logged(self):
        state = dt.State(np.array([1.0, 1.0]), np.eye(2), dt.Epoch(0))
        ekf = dt.ExtendedKalmanFilter(
            state, lambda x: x, lambda x: np.eye(2), np.zeros((2, 2)),
            lambda x: np.zeros(1), lambda x: np.zeros((1, 2)), np.zeros((1, 1)))

        with self.assertLogs("determined", level="ERROR") as logs:
            ekf.update(dt.Observation(np.array([1.0]), dt.Epoch(0)))
        self.assertIn("singular", logs.output[0])

    def test_backward_predict_is_logged(self):
        state = dt.State(np.array([1.0, 1.0]), np.eye(2), dt.Epoch(5))
        ekf = make_ekf(state, np.eye(2), np.eye(2), np.eye(2), np.eye(2))

        with self.assertLogs("determined", level="WARNING") as logs:
            ekf.predict(dt.Epoch(2))
        self.assertIn("Backward", logs.output[0])


if __name__ == "__main__":
    unittest.main()
