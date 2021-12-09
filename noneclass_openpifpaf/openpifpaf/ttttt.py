import functools
import math
import numpy as np


sink1d = np.linspace((4 - 1.0) / 2.0, -(4 - 1.0) / 2.0, num=4, dtype=np.float32)
sink = np.stack((
    sink1d.reshape(1, -1).repeat(4, axis=0),
    sink1d.reshape(-1, 1).repeat(4, axis=1),
), axis=0)

print(sink)

print(np.linalg.norm(sink, axis=0))