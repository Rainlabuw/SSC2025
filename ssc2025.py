from scipy import signal
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.interpolate import InterpolatedUnivariateSpline

#########Global vriabls


Ts = 50  #50 second
dt = 0.1
T = int(Ts/dt) # Total time steps

# Assume symmetric cube
Ixx = 1
Iyy = 1
Izz = 1
I = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])

if __name__ == "__main__":
    gg = 5
