import numpy as np

import determined as dt

from update import update

value = np.array([1.0, 0.0])
epoch = dt.Epoch(0)

state = dt.State(value, covariance=np.eye(value.size), epoch=epoch)
obs = dt.Observation(np.array([0.0, 1.0]), epoch=epoch)

filter = dt.CustomFilter(update)

if __name__ == "__main__":

    new_state = filter.predict(epoch)
    print("Predicted State:")
    print(new_state)

    new_state = filter.update(obs)
    print("Updated State:")
    print(new_state)