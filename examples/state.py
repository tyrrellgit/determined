import numpy as np

import determined as dt

value = np.array([1.0, 0.0])
cov = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

epoch = dt.Epoch(0)
state = dt.State(value, cov, epoch)

print(epoch)
print(state)