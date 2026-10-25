import unittest
import numpy as np

import determined as dt


def make_state(value, cov=None, epoch=0):
    value = np.asarray(value, dtype=float)
    cov = np.eye(value.size) if cov is None else cov
    return dt.State(value, covariance=cov, epoch=dt.Epoch(epoch))


def reference_ukf_step(x, p, f, q, alpha, beta, kappa):
    """Independent numpy unscented transform through f, plus q."""
    n = x.size
    lam = alpha ** 2 * (n + kappa) - n
    l = np.linalg.cholesky((n + lam) * p)
    sigma = np.vstack([x, x + l.T, x - l.T])
    wm = np.full(2 * n + 1, 1 / (2 * (n + lam)))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = wm[0] + 1 - alpha ** 2 + beta
    return sigma, wm, wc


def reference_predict(x, p, f, q, t, params):
    sigma, wm, wc = reference_ukf_step(x, p, f, q, *params)
    y = np.array([f(s, t) for s in sigma])
    mean = wm @ y
    d = y - mean
    return mean, (wc * d.T) @ d + q


def reference_update(x, p, z, h, r, t, params):
    sigma, wm, wc = reference_ukf_step(x, p, None, None, *params)
    zs = np.array([h(s, t) for s in sigma])
    zm = wm @ zs
    dz, dx = zs - zm, sigma - x
    s = (wc * dz.T) @ dz + r
    pxz = (wc * dx.T) @ dz
    k = pxz @ np.linalg.inv(s)
    return x + k @ (z - zm), p - k @ s @ k.T


class TestUnscentedKalmanFilter(unittest.TestCase):

    def test_matches_numpy_reference(self):
        """Nonlinear models, process noise, epoch gaps, non-default parameters."""
        params = (0.5, 2.0, 1.0)
        f = lambda x, t: np.array([x[0] + 0.1 * np.sin(x[1]), 0.95 * x[1] + 0.01 * t])
        h = lambda x, t: np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])
        q, r = np.eye(2) * 0.01, np.diag([0.1, 0.01])

        x, p = np.array([1.0, 0.5]), np.eye(2) * 0.3
        ukf = dt.UnscentedKalmanFilter(
            make_state(x, p), f=f, q=q, h=h, r=r,
            alpha=params[0], beta=params[1], kappa=params[2])

        epoch = 0
        for gap in [1, 2, 1, 3]:
            for _ in range(gap):
                x, p = reference_predict(x, p, f, q, epoch, params)
                epoch += 1
            z = np.array([1.2, 0.3 + 0.01 * epoch])
            x, p = reference_update(x, p, z, h, r, epoch, params)

            out = ukf.update(dt.Observation(z, epoch=dt.Epoch(epoch)))
            np.testing.assert_allclose(out.value, x, atol=1e-10)
            np.testing.assert_allclose(out.covariance, (p + p.T) / 2, atol=1e-10)
            self.assertEqual(out.epoch.value, epoch)

    def test_matches_kalman_filter_on_linear_models(self):
        f_mat = np.array([[1.0, 0.5], [0.0, 1.0]])
        h_mat = np.array([[1.0, 0.0]])
        q, r = np.eye(2) * 0.05, np.array([[0.1]])

        kf = dt.KalmanFilter(make_state([0.0, 1.0]), f=f_mat, q=q, h=h_mat, r=r)
        ukf = dt.UnscentedKalmanFilter(
            make_state([0.0, 1.0]), f=lambda x, t: f_mat @ x, q=q, h=lambda x, t: h_mat @ x, r=r)

        for t in [1, 2, 4, 5]:
            obs = dt.Observation(np.array([0.4 * t]), epoch=dt.Epoch(t))
            a, b = kf.update(obs), ukf.update(obs)
            np.testing.assert_allclose(a.value, b.value, atol=1e-10)
            np.testing.assert_allclose(a.covariance, b.covariance, atol=1e-10)

    def test_transform_recovers_exact_moments_of_square(self):
        """x ~ N(m, s2), f(x) = x^2: mean m^2 + s2, variance 4 m^2 s2 + 2 s2^2."""
        m, s2 = 2.0, 0.5
        ukf = dt.UnscentedKalmanFilter(
            make_state([m], np.array([[s2]])),
            f=lambda x, t: x ** 2, q=np.zeros((1, 1)), h=lambda x, t: x, r=np.eye(1))

        out = ukf.predict(dt.Epoch(1))
        np.testing.assert_allclose(out.value, [m * m + s2])
        np.testing.assert_allclose(out.covariance, [[4 * m * m * s2 + 2 * s2 * s2]])

    def test_models_receive_epoch_at_every_sigma_point(self):
        calls = []

        def f(x, t):
            calls.append(t)
            return x

        ukf = dt.UnscentedKalmanFilter(
            make_state([0.0, 0.0]), f=f, q=np.eye(2), h=lambda x, t: x, r=np.eye(2))
        ukf.predict(dt.Epoch(2))
        self.assertEqual(calls, [0] * 5 + [1] * 5)  # 2n + 1 = 5 sigma points per step

    def test_error_in_callback_leaves_state_unchanged(self):
        def h(x, t):
            raise RuntimeError("boom")

        state = make_state([1.0, 2.0])
        ukf = dt.UnscentedKalmanFilter(
            state, f=lambda x, t: x + 1.0, q=np.eye(2), h=h, r=np.eye(2))

        with self.assertRaises(RuntimeError):
            ukf.update(dt.Observation(np.array([0.0, 0.0]), epoch=dt.Epoch(1)))

        # the predict step inside update was rolled back too
        np.testing.assert_array_equal(state.value, [1.0, 2.0])
        np.testing.assert_array_equal(state.covariance, np.eye(2))
        self.assertEqual(state.epoch.value, 0)

    def test_not_positive_definite_covariance_is_logged(self):
        state = make_state([1.0, 1.0], -np.eye(2))
        ukf = dt.UnscentedKalmanFilter(
            state, f=lambda x, t: 2 * x, q=np.zeros((2, 2)), h=lambda x, t: x, r=np.eye(2))

        with self.assertLogs("determined", level="ERROR") as logs:
            out = ukf.predict(dt.Epoch(1))
        self.assertIn("positive definite", logs.output[0])
        np.testing.assert_array_equal(out.value, [1.0, 1.0])
        self.assertEqual(out.epoch.value, 0)

    def test_reset_and_state_setter(self):
        ukf = dt.UnscentedKalmanFilter(
            make_state([1.0, 1.0], np.eye(2) * 3), f=lambda x, t: x, q=np.eye(2),
            h=lambda x, t: x, r=np.eye(2))

        ukf.reset()
        np.testing.assert_array_equal(ukf.state.value, [0.0, 0.0])
        np.testing.assert_array_equal(ukf.state.covariance, np.eye(2))

        ukf.state = make_state([5.0, 6.0], epoch=3)
        np.testing.assert_array_equal(ukf.state.value, [5.0, 6.0])
        self.assertEqual(ukf.state.epoch.value, 3)

    def test_bad_inputs_raise(self):
        state, eye = make_state([1.0, 1.0]), np.eye(2)

        def build(**overrides):
            args = dict(f=lambda x, t: x, q=eye, h=lambda x, t: x, r=eye)
            args.update(overrides)
            return dt.UnscentedKalmanFilter(state, **args)

        with self.assertRaises(TypeError):
            dt.UnscentedKalmanFilter(state, lambda x, t: x, eye, lambda x, t: x, eye)
        with self.assertRaises(TypeError):
            build(h=None)
        with self.assertRaises(ValueError):
            build(q=np.eye(3))
        with self.assertRaises(ValueError):
            build(f=lambda x, t: np.zeros(3)).predict(dt.Epoch(1))
        with self.assertRaises(ValueError):
            build().update(dt.Observation(np.array([1.0]), epoch=dt.Epoch(1)))


if __name__ == "__main__":
    unittest.main()
