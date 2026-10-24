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


class LinearTransition:
    def __init__(self, f, q, initial_state):
        self.f = f
        self.q = q

        self._state = initial_state
        self._jac = np.zeros((self.f.shape[0], self.f.shape[0]))

    def state(self, epoch: dt.Epoch):

        # state / covariance update
        value = self.f @ self._state.value
        cov = self.f @ self._state.covariance @ self.f.T + self.q
        self._state = dt.State(value, cov, epoch)

        return self._state
    
    def jacobian(self, state: dt.State):
        return self._jac
    

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