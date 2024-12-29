from scipy import signal
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.interpolate import InterpolatedUnivariateSpline


def f(
        x_t: np.ndarray,
        u_t: np.ndarray
) -> np.ndarray:
    omega_t = x_t[0:3]
    q_t = x_t[3:]
    p = np.zeros(4)
    p[1:] = omega_t
    Omega = np.array([[p[0], -p[1], -p[2], -p[3]],
                      [p[1], p[0], p[3], -p[2]],
                      [p[2], -p[3], p[0], p[1]],
                      [p[3], p[2], -p[1], p[0]]])

    omegadot_t = LA.inv(J) @ (u_t - np.cross(omega_t, (J @ omega_t)))
    qdot_t = 0.5 * Omega @ q_t
    # qdot_t = np.zeros(4)

    x_dot = np.concatenate((omegadot_t, qdot_t), 0)

    return x_dot


# From quaternion to rotation matrix (123)
def q2R(
        q: np.ndarray
) -> np.ndarray:
    R = np.array(
        [[1 - 2 * (q[2] * q[2] + q[3] * q[3]), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
         [2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] * q[1] + q[3] * q[3]), 2 * (q[2] * q[3] - q[0] * q[1])],
         [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])]])
    return R


# From quaternion to Euler angle (123)
def q2e(
        q: np.ndarray
) -> np.ndarray:
    R = q2R(q)
    euler_angle = np.array([np.arctan2(R[2, 1], R[2, 2]),
                            -np.arcsin(R[2, 0]),
                            np.arctan2(R[1, 0], R[0, 0])])
    return euler_angle


def attitude_plot(
        ib: np.ndarray,
        jb: np.ndarray,
        kb: np.ndarray
):
    # Create a sphere
    phi, theta = np.linspace(0, np.pi, 20), np.linspace(0, 2 * np.pi, 20)
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Create the 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the sphere
    ax.plot_surface(x, y, z, color='cyan', alpha=0.3, edgecolor='none')

    # Plot the unit vectors
    origin = np.array([0, 0, 0])
    for t in range(T):
        if np.mod(t, 10) == 0:
            i = ib[:, t]
            j = jb[:, t]
            k = kb[:, t]
            ax.plot([i[0]], [i[1]], [i[2]], 'r.')
            i_line, = ax.plot([origin[0], i[0]], [origin[1], i[1]], [origin[2], i[2]], 'r')
            j_line, = ax.plot([origin[0], j[0]], [origin[1], j[1]], [origin[2], j[2]], 'g')
            k_line, = ax.plot([origin[0], k[0]], [origin[1], k[1]], [origin[2], k[2]], 'b')

            # Set the aspect ratio to be equal
            ax.set_box_aspect([1, 1, 1])

            # Set the limits
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])

            ax.set_zlim([1, -1])
            # Add labels and legend
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.pause(0.1)
            i_line.remove()
            j_line.remove()
            k_line.remove()
            # plt.clf()
    plt.show()
    return None


#########Global vriabls


Ts = 50  # 50 second
dt = 0.2
T = int(Ts / dt)  # Total time steps

# Assume symmetric cube
Ixx = 1
Iyy = 1
Izz = 1
J = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])

if __name__ == "__main__":
    q = np.zeros((4, T))
    omega = np.zeros((3, T))
    q[:, 0] = np.array([1, 0, 0, 0])
    x = np.concatenate((omega, q), 0)
    u = np.zeros((3, T - 1))
    u[0, :] = -0.00 * np.ones(T - 1)
    u[2, :] = 0.00 * np.ones(T - 1)
    u[1, :] = 0.001 * np.ones(T - 1)
    p = np.array([0, -0.001, 0.00, 0.00])
    euler = np.zeros((3, T))
    ib = np.zeros((3, T))
    jb = np.zeros((3, T))
    kb = np.zeros((3, T))
    ib[:, 0] = np.array([1, 0, 0])
    jb[:, 0] = np.array([0, 1, 0])
    kb[:, 0] = np.array([0, 0, 1])

    for t in range(T - 1):
        x_t = x[:, t]
        u_t = u[:, t]
        q_t = x_t[3:] / LA.norm(x_t[3:], 2)
        x_dot = f(x_t, u_t)
        x_tp1 = x_t + x_dot * dt
        x[:, t + 1] = x_tp1
        R = q2R(q_t)

        ib[:, t] = ib[:, 0] @ R
        jb[:, t] = jb[:, 0] @ R
        kb[:, t] = kb[:, 0] @ R

        euler[:, t] = q2e(q_t)

    attitude_plot(ib, jb, kb)

    # t = np.linspace(0, 50, T)
    # plt.plot(t, euler[0] * 180 / np.pi, 'r.')
    # plt.plot(t, euler[1] * 180 / np.pi, 'g-')
    # plt.plot(t, euler[2] * 180 / np.pi, 'b.-')
    # plt.legend(['roll', 'pitch', 'yaw'])
    # plt.show()
