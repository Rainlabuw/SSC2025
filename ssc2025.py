from scipy import signal
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.interpolate import InterpolatedUnivariateSpline


def df_dx(
        p: np.ndarray
) -> np.ndarray:
    A_c = np.array([[p[0], -p[1], -p[2], -p[3]],
                    [p[1], p[0], p[3], -p[2]],
                    [p[2], -p[3], p[0], p[1]],
                    [p[3], p[2], -p[1], p[0]]])

    return A_c


def q2R(
        q: np.ndarray
) -> np.ndarray:
    R = np.array([[1 - 2 * (q[2] *q[2] + q[3] * q[3]), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
                  [2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] * q[1] + q[3] * q[3]), 2 * (q[2] * q[3] - q[0] * q[1])],
                  [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])]])
    return R


def q2e(
        q: np.ndarray
) -> np.ndarray:
    R = q2R(q)
    euler_angle = np.array([np.arctan2(R[2, 1], R[2, 2]),
                            -np.arcsin(R[2, 0]),
                            np.arctan2(R[1, 0], R[0, 0])])
    return euler_angle


#########Global vriabls


Ts = 50  # 50 second
dt = 0.1
T = int(Ts / dt)  # Total time steps

# Assume symmetric cube
Ixx = 1
Iyy = 1
Izz = 1
I = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
q = np.zeros((4, T))
if __name__ == "__main__":
    p = np.array([0, -0.001, 0.00, 0.00])
    q[:, 0] = np.array([1, 0, 0, 0])
    euler = np.zeros((3,T))
    for t in range(T - 1):
        A = df_dx(p)
        euler[:, t] = q2e(q[:, t])
        q[:, t + 1] = q[:, t] + A @ q[:, t]
    t = np.linspace(0,50,T)
    plt.plot(t, euler[0],'r.')
    plt.plot(t, euler[1],'g-')
    plt.plot(t, euler[2],'b.-')
    plt.legend(['roll','pitch','yaw'])
    plt.show()