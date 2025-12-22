import numpy as np

import determined as dt

class DummyMeasurement:
    def __init__(self, matrix):
        self.matrix = matrix
        self.m_inv = np.linalg.pinv(matrix)
        self.jac = np.zeros((self.matrix.shape[0], self.matrix.shape[1]))

    def projection(self, state: dt.State):
        z = self.matrix @ state.value
        return dt.Observation(z, state.epoch)

    def inverse(self, observation: dt.Observation):
        x = self.m_inv @ observation.value
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
model = DummyMeasurement(h)
measurement = dt.MeasurementModel(model)

obs = measurement.projection(state)
inv = measurement.inverse(obs)
jac = measurement.jacobian(state)

print(state)
print(obs)
print(inv)
print(jac)