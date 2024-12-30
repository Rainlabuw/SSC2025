from scipy import signal
import jax
import jax.numpy as jnp
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.interpolate import InterpolatedUnivariateSpline

jax.config.update('jax_enable_x64', True)


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


def f_jax(
        x: jnp.ndarray,
        u: jnp.ndarray
) -> jnp.ndarray:
    omega = x[0:3]
    q = x[3:]
    p = jnp.zeros(4)
    # p[1:] = omega
    p = p.at[1:].set(omega)
    Omega = jnp.array([[p[0], -p[1], -p[2], -p[3]],
                       [p[1], p[0], p[3], -p[2]],
                       [p[2], -p[3], p[0], p[1]],
                       [p[3], p[2], -p[1], p[0]]])

    # omegadot = LA.inv(J) @ (u - np.cross(omega, (J @ omega)))
    omegadot = jnp.linalg.inv(J) @ (u - jnp.cross(omega, (J @ omega)))
    qdot = 0.25 * Omega @ q
    # qdot_t = np.zeros(4)


    x_dot_jax = jnp.concatenate((omegadot, qdot), 0)
    return x_dot_jax


def linearize(
        f_jax: jnp.ndarray,
        x_t: np.ndarray,
        u_t: np.ndarray
):
    # Compute the Jacobian of f(x, u) with respect to x (A matrix)
    # A = jax.jacobian(lambda x: f_jax(x, u_t))(x_t)
    A = jax.jacfwd(lambda x: f_jax(x, u_t))(x_t)
    # Compute the Jacobian of f(x, u) with respect to u (B matrix)
    # B = jax.jacobian(lambda u: f_jax(x_t, u))(u_t)
    B = jax.jacfwd(lambda u: f_jax(x_t, u))(u_t)
    return A, B


def discretization(
        A: np.ndarray,
        B: np.ndarray
) -> list:
    C = np.eye(7)
    D = np.zeros((7, 3))
    sys = signal.StateSpace(A, B, C, D)
    sysd = sys.to_discrete(dt)
    Ad = sysd.A
    Bd = sysd.B

    return Ad, Bd


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
            if t < T-10:
                i_line.remove()
                j_line.remove()
                k_line.remove()
            # plt.clf()

    plt.show()
    return None


#########Global vriabls


Ts = 60  # 50 second
dt = 0.1
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
    u[0, :] = -0.002 * np.ones(T - 1)
    u[2, :] = 0.001 * np.ones(T - 1)
    u[1, :] = -0.0013 * np.ones(T - 1)
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
        [A, B] = linearize(f_jax, x_t, u_t)
        A = np.asarray(A)
        B = np.asarray(B)
        # Continuous
        x_dot = f(x_t, u_t)
        # Linearized
        x_dot_ja = A @ x_t + B @ u_t
        # Discretized
        [Ad, Bd] = discretization(A, B)
        # x_tp1 = x_t + x_dot * dt # Continuous
        # x_tp1 = x_t + x_dot_ja * dt # Continuous and linearized
        x_tp1 = Ad @ x_t + Bd @ u_t # Discretized
        x[:, t + 1] = x_tp1
        R = q2R(q_t)

        ib[:, t] = ib[:, 0] @ R
        jb[:, t] = jb[:, 0] @ R
        kb[:, t] = kb[:, 0] @ R

        euler[:, t] = q2e(q_t)
        if np.mod(t, 40) == 0:
            print("x_dot", x_dot)
            print("x_dot_jax", x_dot_ja)

    attitude_plot(ib, jb, kb)

    # t = np.linspace(0, 50, T)
    # plt.plot(t, euler[0] * 180 / np.pi, 'r.')
    # plt.plot(t, euler[1] * 180 / np.pi, 'g-')
    # plt.plot(t, euler[2] * 180 / np.pi, 'b.-')
    # plt.legend(['roll', 'pitch', 'yaw'])
    # plt.show()
