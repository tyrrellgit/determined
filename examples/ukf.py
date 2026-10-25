"""Unscented Kalman filter tracking a target moving at constant velocity with a
range-and-bearing sensor (a radar).

No Jacobians are needed: the filter pushes sigma points through `f` and `h`
instead of linearising them. Both are evaluated at 2N + 1 sigma points per
step, so they must not keep state between calls.
"""
import numpy as np

import determined as dt

# state x = (px, py, vx, vy), one epoch per second
F = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
Q = np.diag([0.01, 0.01, 0.001, 0.001])
R = np.diag([0.1 ** 2, 0.01 ** 2])          # range (m) and bearing (rad) noise


def f(x, t):
    return F @ x                            # constant velocity


def h(x, t):
    return np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])


# the target starts at (10, 5) moving at (-0.5, 0.3); the initial guess is poor
truth = np.array([10.0, 5.0, -0.5, 0.3])
state = dt.State(np.array([9.0, 6.0, 0.0, 0.0]), covariance=np.diag([4.0, 4.0, 1.0, 1.0]),
                 epoch=dt.Epoch(0))

ukf = dt.UnscentedKalmanFilter(state, f=f, q=Q, h=h, r=R)

rng = np.random.default_rng(0)
for t in range(1, 21):
    truth = F @ truth
    z = h(truth, t) + rng.normal(0.0, np.sqrt(np.diag(R)))

    ukf.predict(dt.Epoch(t))
    estimate = ukf.update(dt.Observation(z, epoch=dt.Epoch(t)))

    if t % 5 == 0:
        error = np.linalg.norm(estimate.value[:2] - truth[:2])
        print(f"t={t:2d} position={estimate.value[:2].round(3)} velocity={estimate.value[2:].round(3)} "
              f"position error={error:.3f}")
