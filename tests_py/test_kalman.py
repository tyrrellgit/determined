import unittest
import numpy as np

import determined as dt


def make_state(value, cov=None, epoch=0):
    value = np.asarray(value, dtype=float)
    cov = np.eye(value.size) if cov is None else cov
    return dt.State(value, covariance=cov, epoch=dt.Epoch(epoch))


F = np.array([[1.0, 0.5], [0.0, 1.0]])    # deliberately not symmetric
Q = np.eye(2) * 0.05
H = np.array([[1.0, 0.0]])                 # non-square (1 x 2)
R = np.array([[0.1]])


def make_kf(state=None):
    return dt.KalmanFilter(state or make_state([0.0, 1.0]), f=F, q=Q, h=H, r=R)


class TestKalmanFilter(unittest.TestCase):

    def test_matches_reference(self):
        """Matches a closed-form KF in numpy, including multi-epoch gaps."""
        kf = make_kf()
        x, p = np.array([0.0, 1.0]), np.eye(2)

        epoch = 0
        for gap in [1, 1, 3, 2]:
            epoch += gap
            z = np.array([0.4 * epoch])
            kf.predict(dt.Epoch(epoch))
            out = kf.update(dt.Observation(z, epoch=dt.Epoch(epoch)))

            for _ in range(gap):
                x, p = F @ x, F @ p @ F.T + Q
            s = H @ p @ H.T + R
            k = p @ H.T @ np.linalg.inv(s)
            x, p = x + k @ (z - H @ x), (np.eye(2) - k @ H) @ p

            np.testing.assert_allclose(out.value, x, atol=1e-10)
            np.testing.assert_allclose(out.covariance, p, atol=1e-10)
            self.assertEqual(out.epoch.value, epoch)

    def test_update_propagates_to_observation_epoch(self):
        """update() alone (no predict) brings the state to the observation epoch."""
        with_predict, without = make_kf(), make_kf()
        obs = dt.Observation(np.array([1.0]), epoch=dt.Epoch(3))

        with_predict.predict(dt.Epoch(3))
        a = with_predict.update(obs)
        b = without.update(obs)

        np.testing.assert_allclose(a.value, b.value)
        np.testing.assert_allclose(a.covariance, b.covariance)

    def test_matches_extended_kalman_filter(self):
        kf = make_kf()
        ekf = dt.ExtendedKalmanFilter(
            make_state([0.0, 1.0]),
            f=lambda x, t: F @ x, jac_f=lambda x, t: F, q=Q,
            h=lambda x, t: H @ x, jac_h=lambda x, t: H, r=R)

        for t in range(1, 5):
            obs = dt.Observation(np.array([0.3 * t]), epoch=dt.Epoch(t))
            a, b = kf.update(obs), ekf.update(obs)
            np.testing.assert_allclose(a.value, b.value, atol=1e-12)
            np.testing.assert_allclose(a.covariance, b.covariance, atol=1e-12)

    def test_predict_is_idempotent_and_state_is_shared(self):
        state = make_state([0.0, 1.0])
        kf = make_kf(state)

        first = kf.predict(dt.Epoch(2)).value.copy()
        again = kf.predict(dt.Epoch(2))
        np.testing.assert_allclose(again.value, first)
        np.testing.assert_allclose(first, F @ F @ [0.0, 1.0])

        # zero-copy: the state passed in is the filter's state
        np.testing.assert_array_equal(state.value, first)
        self.assertEqual(state.epoch.value, 2)

    def test_reset_and_state_setter(self):
        kf = make_kf()
        kf.predict(dt.Epoch(1))

        kf.reset()
        np.testing.assert_array_equal(kf.state.value, [0.0, 0.0])
        np.testing.assert_array_equal(kf.state.covariance, np.eye(2))

        kf.state = make_state([5.0, 6.0], epoch=3)
        np.testing.assert_array_equal(kf.state.value, [5.0, 6.0])
        self.assertEqual(kf.state.epoch.value, 3)

        with self.assertRaises(ValueError):
            kf.state = make_state([1.0, 2.0, 3.0])

    def test_arguments_are_keyword_only(self):
        with self.assertRaises(TypeError):
            dt.KalmanFilter(make_state([0.0, 1.0]), F, Q, H, R)

    def test_bad_shapes_raise(self):
        state = make_state([0.0, 1.0])
        good = dict(f=F, q=Q, h=H, r=R)

        for key, bad in [("f", np.eye(3)), ("q", np.eye(3)), ("h", np.ones((1, 3))),
                         ("r", np.eye(2))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                dt.KalmanFilter(state, **{**good, key: bad})

        with self.assertRaises(ValueError):
            make_kf().update(dt.Observation(np.array([1.0, 2.0]), epoch=dt.Epoch(1)))

    def test_backward_predict_inverts_forward(self):
        """Uses the Rust LinearTransition, which propagates backwards with F^-1."""
        state = make_state([0.0, 1.0], np.diag([2.0, 3.0]))
        kf = make_kf(state)
        value, cov = state.value.copy(), state.covariance.copy()

        kf.predict(dt.Epoch(3))
        out = kf.predict(dt.Epoch(0))
        np.testing.assert_allclose(out.value, value, atol=1e-12)
        np.testing.assert_allclose(out.covariance, cov, atol=1e-12)
        self.assertEqual(out.epoch.value, 0)

    def test_singular_update_is_logged(self):
        singular = dt.KalmanFilter(
            make_state([1.0, 1.0]), f=np.eye(2), q=np.zeros((2, 2)),
            h=np.zeros((1, 2)), r=np.zeros((1, 1)))
        with self.assertLogs("determined", level="ERROR") as logs:
            singular.update(dt.Observation(np.array([1.0]), epoch=dt.Epoch(0)))
        self.assertIn("singular", logs.output[0])


class TestCustomFilter(unittest.TestCase):

    def test_is_a_distinct_class_from_kalman_filter(self):
        self.assertIsNot(dt.CustomFilter, dt.KalmanFilter)


if __name__ == "__main__":
    unittest.main()
