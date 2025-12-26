import numpy as np

import determined as dt

class LinearTransition:
    def __init__(self, f, q, initial_state):
        self.f = f
        self.q = q

        self._state = initial_state
        self._jac = np.zeros((self.f.size, self.f.size))

    def state(self, epoch: dt.Epoch):

        # state / covariance update
        value = self.f @ self._state.value
        cov = self.f @ self._state.covariance @ self.f.T + self.q
        self._state = dt.State(value, cov, epoch)

        return self._state
    
    def jacobian(self, state: dt.State):
        return self._jac

value = np.array([1.0, 0.0])
cov = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

epoch = dt.Epoch(0)
state = dt.State(value, cov, epoch)
initial_state = dt.State(value, cov, epoch)

f = np.array([1.5, 2.0])
q = np.eye(f.size)

# coupling of initial_state and state is a poor design choice
# --> results in deadlocks as TransitionModel needs to lock and mutate state
model = LinearTransition(f, q, initial_state)
transition = dt.TransitionModel(model, state)

if __name__ == "__main__":
    for n in range(2):
        _e = dt.Epoch(n)
        transition.state(_e)
        print(state)

    jac = transition.jacobian(state)
    print(jac)