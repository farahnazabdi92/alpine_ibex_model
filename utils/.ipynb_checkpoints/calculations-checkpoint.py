import numpy as np
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def distance(a, b): return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))
