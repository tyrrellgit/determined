"""UD Kalman filter tracking a target moving at constant velocity from noisy,
correlated position measurements, alongside the standard Kalman filter.

The UD filter holds the covariance as factors P = U D U^T and propagates and
updates the factors directly, so P stays symmetric positive semi-definite by
construction, which matters for long runs and badly scaled problems. It gives
the same estimates as `KalmanFilter`, as this example checks.
"""
import numpy as np

import determined as dt

# state x = (px, py, vx, vy), one epoch per second, position measured
F = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
Q = np.diag([0.01, 0.01, 0.001, 0.001])
H = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
])
R = np.array([
    [0.04, 0.03],                           # correlated noise: decorrelated internally
    [0.03, 0.09],
])


def initial_state():
    return dt.State(np.zeros(4), covariance=np.diag([10.0, 10.0, 1.0, 1.0]), epoch=dt.Epoch(0))


ud = dt.UDKalmanFilter(initial_state(), f=F, q=Q, h=H, r=R)
kf = dt.KalmanFilter(initial_state(), f=F, q=Q, h=H, r=R)

truth = np.array([0.0, 0.0, 1.0, 0.5])
rng = np.random.default_rng(0)
noise = np.linalg.cholesky(R)

for t in range(1, 31):
    truth = F @ truth
    z = H @ truth + noise @ rng.normal(size=2)
    obs = dt.Observation(z, epoch=dt.Epoch(t))

    a, b = ud.update(obs), kf.update(obs)

    if t % 10 == 0:
        print(f"t={t:2d} estimate={a.value.round(3)} truth={truth.round(3)}")
        print(f"      |UD - KF| state={np.abs(a.value - b.value).max():.1e} "
              f"covariance={np.abs(a.covariance - b.covariance).max():.1e}")

p = ud.state.covariance
print(f"P symmetric: {np.array_equal(p, p.T)}, smallest eigenvalue: {np.linalg.eigvalsh(p).min():.2e}")
