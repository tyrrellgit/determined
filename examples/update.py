import numpy as np

import determined as dt

from transition import transition
from measurement import measurement

class DummyUpdate:
    def __init__(self, t, m):
        self.t = t
        self.m = m

    def state(self, epoch: dt.Epoch):
        return self.t.state(epoch)

    def apply(self, observation: dt.Observation):
        return self.m.inverse(observation)

    def jacobian(self, state: dt.State):
        jac = self.t.jacobian(state)
        return jac

value = np.array([1.0, 0.0])
epoch = dt.Epoch(0)

state = dt.State(value, np.eye(value.size), epoch)
obs = dt.Observation(np.array([0.0, 1.0]), epoch)

model = DummyUpdate(transition, measurement)
update = dt.UpdateModel(model, state)


if __name__ == "__main__":

    future_state = update.state(epoch)
    new_state = update.apply(obs)
    jac = update.jacobian(state)

    print(future_state)
    print(new_state)
    print(jac)