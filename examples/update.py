import numpy as np

import determined as dt

from transition import transition
from measurement import measurement

class DummyUpdate:
    def __init__(self, t, m):
        self.t = t
        self.m = m

    def apply(self, observation: dt.Observation):
        return self.m.inverse(observation)

    def jacobian(self, state: dt.State):
        jac = self.t.jacobian(state)
        return jac

value = np.array([1.0, 0.0])
epoch = dt.Epoch(0)

state = dt.State(value, np.eye(value.size), epoch)
obs = dt.Observation(np.array([1.0, 0.0]), epoch)

model = DummyUpdate(transition, measurement)
update = dt.UpdateModel(model, state)

new_state = update.apply(obs)
jac = update.jacobian(state)

print(new_state)
print(jac)