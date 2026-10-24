"""Extended Kalman filter tracking a static 2D target from range-only measurements.

The model functions take and return plain numpy arrays, so the Jacobians could
equally come from an autodiff library, e.g. `jac_h = jax.jacfwd(h)`.
"""
import numpy as np

import determined as dt

# initial guess (2, 2) with a wide covariance; the target is really at (3, 4)
state = dt.State(np.array([2.0, 2.0]), np.eye(2) * 4.0, dt.Epoch(0))


def f(x):
    return x                      # static target


def jac_f(x):
    return np.eye(2)


def h(x):
    return np.array([np.linalg.norm(x)])


def jac_h(x):
    return (x / np.linalg.norm(x))[None, :]


ekf = dt.ExtendedKalmanFilter(
    state,
    f=f, jac_f=jac_f, q=np.zeros((2, 2)),
    h=h, jac_h=jac_h, r=np.array([[0.01]]),
)

for t in range(1, 6):
    ekf.predict(dt.Epoch(t))
    estimate = ekf.update(dt.Observation(np.array([5.0]), dt.Epoch(t)))
    print(f"t={t} x={estimate.value} range={np.linalg.norm(estimate.value):.4f}")
