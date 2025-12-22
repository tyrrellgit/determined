import numpy as np

import determined as dt

class DummyTransition:
    def __init__(self, value):
        self.value = value
        self.jac = np.zeros((self.value.size, self.value.size))

    def state(self, epoch: dt.Epoch):
        self.value += epoch.value
        cov = np.eye(self.value.size)
        return dt.State(self.value, cov, epoch)
    
    def jacobian(self, state: dt.State):
        return self.jac

value = np.array([1.0, 0.0])
cov = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

epoch = dt.Epoch(0)
state = dt.State(value, cov, epoch)

model = DummyTransition(state.value)
transition = dt.TransitionModel(model, state)

if __name__ == "__main__":
    for n in range(2):
        _e = dt.Epoch(n)
        transition.state(_e)
        print(state)

    jac = transition.jacobian(state)
    print(jac)