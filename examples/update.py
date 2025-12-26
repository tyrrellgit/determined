import numpy as np

import determined as dt

from transition import transition
from measurement import measurement

class LinearUpdate:
    def __init__(self, t, m):
        self.t = t
        self.m = m

    def apply(self, observation: dt.Observation):

        # propagate state to epoch
        _state = self.t.state(observation.epoch)
        _cov = _state.covariance
        epoch = _state.epoch

        # project observation to state
        z_x = self.m.projection(_state).value
        z = observation.value

        # compute gains
        h = self.m.model.h
        h_t = h.transpose()
        r = self.m.model.r
        s = h @ _cov @ h_t + r
        s_inv = np.linalg.pinv(s)

        # compute innovation update
        k = _cov @ h_t @ s_inv
        y = z - z_x

        value = _state.value + k @ y
        cov = (np.eye(value.size) - k @ h) @ _cov

        return dt.State(value, cov, epoch)

    def jacobian(self, state: dt.State):
        jac = self.t.jacobian(state)
        return jac

value = np.array([1.0, 0.0])
epoch = dt.Epoch(0)

state = dt.State(value, np.eye(value.size), epoch)
obs = dt.Observation(np.array([0.0, 1.0]), epoch)

model = LinearUpdate(transition, measurement)

update = dt.UpdateModel(model, transition, measurement)


if __name__ == "__main__":

    future_state = update.state(epoch)
    new_state = update.apply(obs)
    jac = update.jacobian(state)

    print(future_state)
    print(new_state)
    print(jac)