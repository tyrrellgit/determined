import unittest
import numpy as np

import determined as dt


def make_state(value, cov=None, epoch=0):
    value = np.asarray(value, dtype=float)
    cov = np.eye(value.size) if cov is None else cov
    return dt.State(value, covariance=cov, epoch=dt.Epoch(epoch))


P0 = np.array([[2.0, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 0.5]])
F = np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.1], [0.0, 0.0, 0.95]])
Q = np.array([[0.02, 0.005, 0.0], [0.005, 0.01, 0.002], [0.0, 0.002, 0.03]])
H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
R_CORRELATED = np.array([[0.1, 0.04], [0.04, 0.2]])


class TestUDKalmanFilter(unittest.TestCase):

    def test_matches_kalman_filter(self):
        """Same estimates as KalmanFilter, for diagonal and correlated R."""
        for r in [np.diag([0.1, 0.2]), R_CORRELATED]:
            kf = dt.KalmanFilter(make_state([1.0, -0.5, 2.0], P0), f=F, q=Q, h=H, r=r)
            ud = dt.UDKalmanFilter(make_state([1.0, -0.5, 2.0], P0), f=F, q=Q, h=H, r=r)

            for t in [1, 2, 5, 6, 9]:
                obs = dt.Observation(np.array([0.3 * t, -0.1 * t]), epoch=dt.Epoch(t))
                a, b = kf.update(obs), ud.update(obs)
                np.testing.assert_allclose(b.value, a.value, rtol=1e-9, atol=1e-12)
                np.testing.assert_allclose(b.covariance, a.covariance, rtol=1e-9, atol=1e-12)
                self.assertEqual(a.epoch.value, b.epoch.value)

    def test_covariance_is_symmetric(self):
        state = make_state([0.0, 0.0], np.diag([1e8, 1.0]))
        ud = dt.UDKalmanFilter(state, f=np.eye(2), q=np.zeros((2, 2)),
                               h=np.array([[1.0, 1.0]]), r=np.array([[1e-8]]))
        for t in range(1, 50):
            cov = ud.update(dt.Observation(np.array([1.0]), epoch=dt.Epoch(t))).covariance
            np.testing.assert_array_equal(cov, cov.T)
            self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(), -1e-9 * np.abs(cov).max())

    def test_state_setter_refactorises(self):
        """After the state is replaced, the filter continues from it like a fresh KalmanFilter."""
        ud = dt.UDKalmanFilter(make_state([1.0, -0.5, 2.0], P0), f=F, q=Q, h=H, r=R_CORRELATED)
        ud.update(dt.Observation(np.array([1.0, 1.0]), epoch=dt.Epoch(1)))

        ud.state = make_state([0.5, 0.5, 0.5], np.diag([3.0, 2.0, 1.0]), epoch=1)
        kf = dt.KalmanFilter(make_state([0.5, 0.5, 0.5], np.diag([3.0, 2.0, 1.0]), epoch=1),
                             f=F, q=Q, h=H, r=R_CORRELATED)

        obs = dt.Observation(np.array([0.2, -0.3]), epoch=dt.Epoch(3))
        a, b = kf.update(obs), ud.update(obs)
        np.testing.assert_allclose(b.value, a.value, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(b.covariance, a.covariance, rtol=1e-9, atol=1e-12)

    def test_backward_predict_is_not_supported(self):
        state = make_state([1.0, -0.5, 2.0], P0, epoch=5)
        ud = dt.UDKalmanFilter(state, f=F, q=Q, h=H, r=R_CORRELATED)
        with self.assertLogs("determined", level="WARNING") as logs:
            out = ud.predict(dt.Epoch(2))
        self.assertIn("Backward", logs.output[0])
        self.assertEqual(out.epoch.value, 5)

    def test_reset(self):
        ud = dt.UDKalmanFilter(make_state([1.0, -0.5, 2.0], P0), f=F, q=Q, h=H, r=R_CORRELATED)
        ud.reset()
        np.testing.assert_array_equal(ud.state.value, np.zeros(3))
        np.testing.assert_array_equal(ud.state.covariance, np.eye(3))
        # continues from the reset state
        kf = dt.KalmanFilter(make_state(np.zeros(3)), f=F, q=Q, h=H, r=R_CORRELATED)
        obs = dt.Observation(np.array([0.2, -0.3]), epoch=dt.Epoch(1))
        np.testing.assert_allclose(ud.update(obs).value, kf.update(obs).value, rtol=1e-9, atol=1e-12)

    def test_bad_inputs_raise(self):
        state = make_state([1.0, -0.5, 2.0], P0)
        good = dict(f=F, q=Q, h=H, r=R_CORRELATED)
        with self.assertRaises(TypeError):
            dt.UDKalmanFilter(state, F, Q, H, R_CORRELATED)
        for key, bad in [("f", np.eye(2)), ("q", np.eye(2)), ("h", np.ones((2, 2))), ("r", np.eye(3))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                dt.UDKalmanFilter(state, **{**good, key: bad})
        with self.assertRaises(ValueError):
            dt.UDKalmanFilter(state, **good).update(dt.Observation(np.array([1.0]), epoch=dt.Epoch(1)))


def nonlinear_models():
    f = lambda x, t: np.array([x[0] + 0.1 * np.sin(x[1]), 0.98 * x[1]])
    jac_f = lambda x, t: np.array([[1.0, 0.1 * np.cos(x[1])], [0.0, 0.98]])
    h = lambda x, t: np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])

    def jac_h(x, t):
        r2 = x @ x
        r = np.sqrt(r2)
        return np.array([[x[0] / r, x[1] / r], [-x[1] / r2, x[0] / r2]])

    return dict(f=f, jac_f=jac_f, q=np.eye(2) * 0.01, h=h, jac_h=jac_h,
                r=np.array([[0.05, 0.01], [0.01, 0.02]]))


class TestUDExtendedKalmanFilter(unittest.TestCase):

    def test_matches_extended_kalman_filter(self):
        p = np.array([[1.0, 0.2], [0.2, 1.0]])
        ekf = dt.ExtendedKalmanFilter(make_state([2.0, 2.0], p), **nonlinear_models())
        ud = dt.UDExtendedKalmanFilter(make_state([2.0, 2.0], p), **nonlinear_models())

        for t in [1, 2, 4, 5]:
            obs = dt.Observation(np.array([3.0, 0.8]), epoch=dt.Epoch(t))
            a, b = ekf.update(obs), ud.update(obs)
            np.testing.assert_allclose(b.value, a.value, rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(b.covariance, a.covariance, rtol=1e-9, atol=1e-12)

    def test_error_in_callback_leaves_state_unchanged_and_filter_usable(self):
        models = nonlinear_models()
        calls = {"n": 0}
        h_ok = models["h"]

        def h(x, t):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            return h_ok(x, t)

        p = np.array([[1.0, 0.2], [0.2, 1.0]])
        state = make_state([2.0, 2.0], p)
        ud = dt.UDExtendedKalmanFilter(state, **{**models, "h": h})
        ekf = dt.ExtendedKalmanFilter(make_state([2.0, 2.0], p), **models)

        obs1 = dt.Observation(np.array([3.0, 0.8]), epoch=dt.Epoch(1))
        ud.update(obs1)
        ekf.update(obs1)
        before = (state.value.copy(), state.covariance.copy(), state.epoch.value)

        with self.assertRaises(RuntimeError):
            ud.update(dt.Observation(np.array([3.1, 0.7]), epoch=dt.Epoch(2)))
        np.testing.assert_array_equal(state.value, before[0])
        np.testing.assert_array_equal(state.covariance, before[1])
        self.assertEqual(state.epoch.value, before[2])

        # the factors were restored with the state: it continues like the EKF
        obs3 = dt.Observation(np.array([3.2, 0.75]), epoch=dt.Epoch(3))
        np.testing.assert_allclose(ud.update(obs3).value, ekf.update(obs3).value, rtol=1e-9, atol=1e-12)

    def test_arguments_are_keyword_only(self):
        m = nonlinear_models()
        with self.assertRaises(TypeError):
            dt.UDExtendedKalmanFilter(make_state([2.0, 2.0]), m["f"], m["jac_f"], m["q"],
                                      m["h"], m["jac_h"], m["r"])


if __name__ == "__main__":
    unittest.main()
