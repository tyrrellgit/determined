import unittest
import numpy as np

import determined as dt


def make_ekf(state, f_mat, q, h_mat, r):
    return dt.ExtendedKalmanFilter(
        state,
        f=lambda x, t: f_mat @ x,
        jac_f=lambda x, t: f_mat,
        q=q,
        h=lambda x, t: h_mat @ x,
        jac_h=lambda x, t: h_mat,
        r=r,
    )


def make_state(value, cov=None, epoch=0):
    value = np.asarray(value, dtype=float)
    cov = np.eye(value.size) if cov is None else cov
    return dt.State(value, covariance=cov, epoch=dt.Epoch(epoch))


class TestExtendedKalmanFilter(unittest.TestCase):

    def test_predict(self):
        state = make_state([1.0, 1.0])
        f = np.diag([2.0, 3.0])
        ekf = make_ekf(state, f, np.zeros((2, 2)), np.eye(2), np.eye(2))

        out = ekf.predict(dt.Epoch(2))
        np.testing.assert_allclose(out.value, [4.0, 9.0])
        np.testing.assert_allclose(out.covariance, np.diag([16.0, 81.0]))
        self.assertEqual(out.epoch.value, 2)

        # zero-copy: shares storage with the state it was built from
        np.testing.assert_array_equal(state.value, out.value)

        # predicting to the same epoch again must not propagate
        out = ekf.predict(dt.Epoch(2))
        np.testing.assert_allclose(out.value, [4.0, 9.0])

    def test_returned_states_are_live_views(self):
        """Documented behaviour: results change on the next call."""
        state = make_state([1.0, 1.0])
        ekf = make_ekf(state, np.diag([2.0, 2.0]), np.zeros((2, 2)), np.eye(2), np.eye(2))

        first = ekf.predict(dt.Epoch(1))
        kept = first.value.copy()
        ekf.predict(dt.Epoch(2))

        np.testing.assert_allclose(kept, [2.0, 2.0])
        np.testing.assert_allclose(first.value, [4.0, 4.0])

    def test_matches_reference_kalman_filter(self):
        """A linear problem must match a closed-form KF written in numpy."""
        f = np.array([[1.0, 0.5], [0.0, 1.0]])   # deliberately not symmetric
        q = np.eye(2) * 0.05
        h = np.array([[1.0, 0.0]])                # non-square (1 x 2)
        r = np.array([[0.1]])
        value, cov = np.array([0.0, 1.0]), np.eye(2)

        ekf = make_ekf(make_state(value, cov), f, q, h, r)

        x, p = value.copy(), cov.copy()
        for t in range(1, 5):
            z = np.array([0.4 * t])
            ekf.predict(dt.Epoch(t))
            out = ekf.update(dt.Observation(z, epoch=dt.Epoch(t)))

            x, p = f @ x, f @ p @ f.T + q
            s = h @ p @ h.T + r
            k = p @ h.T @ np.linalg.inv(s)
            x, p = x + k @ (z - h @ x), (np.eye(2) - k @ h) @ p

            np.testing.assert_allclose(out.value, x, atol=1e-12)
            np.testing.assert_allclose(out.covariance, p, atol=1e-12)

    def test_covariance_stays_symmetric_positive(self):
        """Joseph form: badly scaled problem must keep P symmetric and PSD."""
        state = make_state([0.0, 0.0], np.diag([1e8, 1.0]))
        h = np.array([[1.0, 1.0]])
        ekf = make_ekf(state, np.eye(2), np.zeros((2, 2)), h, np.array([[1e-8]]))

        for t in range(1, 50):
            out = ekf.update(dt.Observation(np.array([1.0]), epoch=dt.Epoch(t)))
            cov = out.covariance
            np.testing.assert_allclose(cov, cov.T, rtol=0, atol=1e-6)
            self.assertGreaterEqual(np.linalg.eigvalsh((cov + cov.T) / 2).min(),
                                    -1e-9 * np.abs(cov).max())

    def test_range_sensor(self):
        """Nonlinear range sensor pulls the estimate to the truth."""
        truth_range = 5.0
        state = make_state([2.0, 2.0], np.eye(2) * 4.0)
        ekf = dt.ExtendedKalmanFilter(
            state,
            f=lambda x, t: x,
            jac_f=lambda x, t: np.eye(2),
            q=np.zeros((2, 2)),
            h=lambda x, t: np.array([np.linalg.norm(x)]),
            jac_h=lambda x, t: (x / np.linalg.norm(x))[None, :],
            r=np.array([[0.01]]),
        )

        err = abs(np.linalg.norm(state.value) - truth_range)
        trace = np.trace(state.covariance)
        for t in range(1, 6):
            out = ekf.update(dt.Observation(np.array([truth_range]), epoch=dt.Epoch(t)))
            new_err = abs(np.linalg.norm(out.value) - truth_range)
            new_trace = np.trace(out.covariance)
            self.assertLess(new_err, err)
            self.assertLess(new_trace, trace)
            err, trace = new_err, new_trace
        self.assertLess(err, 1e-2)

    def test_models_receive_epoch(self):
        """f / jac_f get the epoch the step starts at, h / jac_h the measurement epoch."""
        f_epochs, h_epochs = [], []

        def f(x, t):
            f_epochs.append(t)
            return x + t                      # time-varying drift

        def h(x, t):
            h_epochs.append(t)
            return x * t

        state = make_state([0.0])
        ekf = dt.ExtendedKalmanFilter(
            state,
            f=f, jac_f=lambda x, t: np.eye(1), q=np.zeros((1, 1)),
            h=h, jac_h=lambda x, t: np.eye(1) * t, r=np.eye(1),
        )

        out = ekf.predict(dt.Epoch(3))        # steps from t=0, 1, 2
        self.assertEqual(f_epochs, [0, 1, 2])
        np.testing.assert_allclose(out.value, [0 + 1 + 2])

        ekf.update(dt.Observation(np.array([1.0]), epoch=dt.Epoch(3)))
        self.assertEqual(h_epochs, [3])

    def test_callables_may_return_array_likes(self):
        """Lists / (N, 1) arrays are coerced, as AD libraries return their own types."""
        state = make_state([1.0, 2.0])
        ekf = dt.ExtendedKalmanFilter(
            state,
            f=lambda x, t: [float(x[0]), float(x[1])],
            jac_f=lambda x, t: [[1, 0], [0, 1]],
            q=np.zeros((2, 2)),
            h=lambda x, t: x.reshape(2, 1),
            jac_h=lambda x, t: np.eye(2),
            r=np.eye(2),
        )
        out = ekf.update(dt.Observation(np.array([1.0, 2.0]), epoch=dt.Epoch(1)))
        np.testing.assert_allclose(out.value, [1.0, 2.0])

    def test_reset_and_state_setter(self):
        state = make_state([1.0, 1.0], np.eye(2) * 3.0)
        ekf = make_ekf(state, np.eye(2), np.eye(2), np.eye(2), np.eye(2))

        ekf.reset()
        np.testing.assert_array_equal(ekf.state.value, [0.0, 0.0])
        np.testing.assert_array_equal(ekf.state.covariance, np.eye(2))

        ekf.state = make_state([5.0, 6.0], epoch=3)
        np.testing.assert_array_equal(ekf.state.value, [5.0, 6.0])
        self.assertEqual(ekf.state.epoch.value, 3)

    def test_arguments_are_keyword_only(self):
        eye = np.eye(2)
        with self.assertRaises(TypeError):
            dt.ExtendedKalmanFilter(
                make_state([1.0, 1.0]), lambda x, t: x, lambda x, t: eye, eye,
                lambda x, t: x, lambda x, t: eye, eye)

    def test_bad_inputs_raise(self):
        state = make_state([1.0, 1.0])
        eye = np.eye(2)

        def build(**overrides):
            args = dict(f=lambda x, t: x, jac_f=lambda x, t: eye, q=eye,
                        h=lambda x, t: x, jac_h=lambda x, t: eye, r=eye)
            args.update(overrides)
            return dt.ExtendedKalmanFilter(state, **args)

        with self.assertRaises(TypeError):
            build(f=1)
        with self.assertRaises(ValueError):
            build(q=np.eye(3))

        # Jacobian of the wrong shape surfaces as ValueError, not a panic
        with self.assertRaises(ValueError):
            build(jac_f=lambda x, t: np.eye(3)).predict(dt.Epoch(1))

        # wrong observation size
        with self.assertRaises(ValueError):
            build().update(dt.Observation(np.array([1.0]), epoch=dt.Epoch(1)))

        # errors raised inside the callable propagate
        def boom(x, t):
            raise RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            build(f=boom).predict(dt.Epoch(1))


class TestKeywordOnlyConstructors(unittest.TestCase):

    def test_positional_arguments_after_the_first_are_rejected(self):
        value, cov, epoch = np.zeros(2), np.eye(2), dt.Epoch(0)
        with self.assertRaises(TypeError):
            dt.State(value, cov, epoch)
        with self.assertRaises(TypeError):
            dt.Observation(value, epoch)
        dt.State(value, covariance=cov, epoch=epoch)
        dt.Observation(value, epoch=epoch)


class TestMatrixLayout(unittest.TestCase):
    """Round trips of non-square / non-symmetric matrices through the bindings.

    These pass regardless of the internal layout; the non-square `H` in
    `test_matches_reference_kalman_filter` is what checks the layout is correct.
    """

    def test_nonsquare_jacobian_roundtrip(self):
        jac = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        class Measurement:
            def projection(self, s): return dt.Observation(np.zeros(2), epoch=s.epoch)
            def inverse(self, o): return dt.State(np.zeros(3), covariance=np.eye(3), epoch=o.epoch)
            def jacobian(self, s): return jac

        model = dt.MeasurementModel(Measurement())
        state = make_state(np.zeros(3))
        np.testing.assert_array_equal(model.jacobian(state), jac)

    def test_covariance_roundtrip(self):
        cov = np.array([[1.0, 2.0], [3.0, 4.0]])
        state = make_state(np.zeros(2), cov)
        np.testing.assert_array_equal(state.covariance, cov)


class TestLogging(unittest.TestCase):
    """Rust-side log messages arrive through Python's `logging`."""

    def test_singular_update_is_logged(self):
        ekf = dt.ExtendedKalmanFilter(
            make_state([1.0, 1.0]),
            f=lambda x, t: x, jac_f=lambda x, t: np.eye(2), q=np.zeros((2, 2)),
            h=lambda x, t: np.zeros(1), jac_h=lambda x, t: np.zeros((1, 2)),
            r=np.zeros((1, 1)))

        with self.assertLogs("determined", level="ERROR") as logs:
            ekf.update(dt.Observation(np.array([1.0]), epoch=dt.Epoch(0)))
        self.assertIn("singular", logs.output[0])

    def test_backward_predict_is_logged(self):
        state = make_state([1.0, 1.0], epoch=5)
        ekf = make_ekf(state, np.eye(2), np.eye(2), np.eye(2), np.eye(2))

        with self.assertLogs("determined", level="WARNING") as logs:
            ekf.predict(dt.Epoch(2))
        self.assertIn("Backward", logs.output[0])


if __name__ == "__main__":
    unittest.main()
