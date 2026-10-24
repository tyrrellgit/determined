import numpy as np

import determined as dt

class LinearMeasurement:
    def __init__(self, h, r):
        self.h = h
        self.r = r
        self.h_inv = np.linalg.pinv(h)
        self.jac = np.zeros((self.h.shape[0], self.h.shape[1]))

    def projection(self, state: dt.State):
        z = self.h @ state.value
        return dt.Observation(z, state.epoch)

    def inverse(self, observation: dt.Observation):
        x = self.h_inv @ observation.value
        cov = np.eye(x.size)
        return dt.State(x, cov, observation.epoch)

    def jacobian(self, state: dt.State):
        return self.jac


value = np.array([1.0, 0.0])
epoch = dt.Epoch(0)
state = dt.State(value, np.eye(value.size), epoch)

h = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])
r = np.eye(h.shape[0])

model = LinearMeasurement(h, r)
measurement = dt.MeasurementModel(model)

if __name__ == "__main__":
    obs = measurement.projection(state)
    inv = measurement.inverse(obs)
    jac = measurement.jacobian(state)

    print(state)
    print(obs)
    print(inv)
    print(jac)