import numpy as np

import determined as dt

value = np.array([1.0, 0.0])
cov = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

epoch = dt.Epoch(0)
state = dt.State(value, cov, epoch)

class DummyTransition:
    def __init__(self, value):
        self.value = value
        self.jac = np.zeros((self.value.size, self.value.size))

    def state(self, epoch: dt.Epoch):
        self.value += epoch.value
        return self.value
    
    def jacobian(self, state: dt.State):
        return self.jac

model = DummyTransition(state.value)
transition = dt.TransitionModel(model, state)

for n in range(2):
    _e = dt.Epoch(n)
    transition.state(_e)
    print(state.value)

jac = transition.jacobian(state)
print(jac)