import numpy as np
import sys


filename = sys.argv[1]
print(filename)
data = np.load(filename)
print(data)
