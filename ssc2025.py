from scipy import signal
import jax
import jax.numpy as jnp
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.interpolate import InterpolatedUnivariateSpline

jax.config.update('jax_enable_x64', True)


def f_continuous(
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


def f_discretized(
        x_t: np.ndarray,
        u_t: np.ndarray
) -> np.ndarray:
    x_dot = f_continuous(x_t, u_t)
    x_tp1 = x_t + x_dot * dt
    return x_tp1


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


def sub_problem_cost_fun(
        lambda_param: float,
        w: cp.Variable,
        v: cp.Variable,
        U_traj: np.ndarray
):
    sub_problem_cost = lambda_param * cp.norm(((U_traj + w)), 1) + 1 * lambda_param * cp.sum(
        cp.sum(cp.abs(v)))
    return sub_problem_cost


def solve_convex_optimal_control_subproblem(
        X_traj: np.ndarray,
        U_traj: np.ndarray,
        x_des: np.ndarray,
) -> list:
    r = 0.5
    lambda_param = 10000
    # Define variables for optimization
    w = cp.Variable((3, T - 1))
    v = cp.Variable((n, T - 1))
    d = cp.Variable((n, T))
    constraints = [d[:, 0] == np.zeros(7)]
    E = np.eye(7)
    sup_problem_cost = sub_problem_cost_fun(lambda_param, w, v, U_traj)
    for t in range(T - 1):
        x_t = X_traj[:, t]
        x_tp1 = X_traj[:, t + 1]
        d_t = d[:, t]
        d_tp1 = d[:, t + 1]
        u_t = U_traj[:, t]
        w_t = w[:, t]
        v_t = v[:, t]
        [A, B] = linearize(f_jax, x_t, u_t)
        [Ad, Bd] = discretization(A, B)
        constraints.append(
            x_tp1 + d_tp1 ==
            f_discretized(x_t, u_t) +
            Ad @ d_t + Bd @ w_t + E @ v_t)
        constraints.append(cp.abs(w_t) <= r)
    # Terminal condition
    constraints.append(X_traj[:, T - 1] + d[:, T - 1] == np.array(
        [x_des[0], x_des[1], x_des[2], x_des[3], -x_des[4], -x_des[5], -x_des[6]]))

    # Define the problem
    problem = cp.Problem(cp.Minimize(sup_problem_cost), constraints)
    problem.solve(solver=cp.CLARABEL)
    cost = problem.value
    w_traj_val = w.value
    d_traj_val = d.value
    return cost, d_traj_val, w_traj_val


def tra_gen(
        X_traj: np.ndarray,
        U_traj: np.ndarray,
        x_des: np.ndarray
) -> list:
    iter = 2

    for i in range(iter):
        [cost, d_traj_val, w_traj_val] = solve_convex_optimal_control_subproblem(X_traj, U_traj, x_des)
        X_traj = X_traj + d_traj_val
        U_traj = U_traj + w_traj_val
        print(cost)

    return X_traj, U_traj


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


# Take in euler angle in degree
def e2q(
        euler: np.ndarray
) -> np.ndarray:
    euler = euler * np.pi / 180
    q0 = np.cos(euler[0] / 2) * np.cos(euler[1] / 2) * np.cos(euler[2] / 2) + np.sin(euler[0] / 2) * np.sin(
        euler[1] / 2) * np.sin(euler[2] / 2)
    q1 = np.sin(euler[0] / 2) * np.cos(euler[1] / 2) * np.cos(euler[2] / 2) - np.cos(euler[0] / 2) * np.sin(
        euler[1] / 2) * np.sin(euler[2] / 2)
    q2 = np.cos(euler[0] / 2) * np.sin(euler[1] / 2) * np.cos(euler[2] / 2) + np.sin(euler[0] / 2) * np.cos(
        euler[1] / 2) * np.sin(euler[2] / 2)
    q3 = np.cos(euler[0] / 2) * np.cos(euler[1] / 2) * np.sin(euler[2] / 2) - np.sin(euler[0] / 2) * np.sin(
        euler[1] / 2) * np.cos(euler[2] / 2)
    q_out = np.array([q0, -q1, -q2, -q3])
    q_out = q_out / LA.norm(q_out, 2)
    return q_out


def attitude_plot(
        euler_des: np.ndarray,
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

    # Plot desired attitude
    origin = np.array([0, 0, 0])
    ib_des = np.array([1, 0, 0])
    jb_des = np.array([0, 1, 0])
    kb_des = np.array([0, 0, 1])
    q_des = e2q(euler_des)
    R_des = q2R(q_des)
    ib_des = R_des @ ib_des
    jb_des = R_des @ jb_des
    kb_des = R_des @ kb_des
    ax.plot([origin[0], ib_des[0]], [origin[1], ib_des[1]], [origin[2], ib_des[2]], 'r')
    ax.plot([origin[0], jb_des[0]], [origin[1], jb_des[1]], [origin[2], jb_des[2]], 'g')
    ax.plot([origin[0], kb_des[0]], [origin[1], kb_des[1]], [origin[2], kb_des[2]], 'b')

    # Plot the initial and desired attitude
    ax.plot([origin[0], ib_des[0]], [origin[1], ib_des[1]], [origin[2], ib_des[2]], 'r')
    ax.plot([ib_des[0]], [ib_des[1]], [ib_des[2]], 'co')

    # Plot the keep out cone
    x_axis = np.array([1, 0, 0])
    zone_att_0 = np.array([0, keep_out_att[1] + half_angle, keep_out_att[2]])
    zone_q_0 = e2q(zone_att_0)
    zone_R_0 = q2R(zone_q_0)
    zone_vec_0 = zone_R_0 @ x_axis
    zone_q_center = e2q(keep_out_att)
    zone_vec_center = q2R(zone_q_center) @ x_axis
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot([zone_vec_center[0]], [zone_vec_center[1]], [zone_vec_center[2]], 'co')
    for i in range(200):
        theta_i = theta[i]
        q0 = np.array([np.cos(theta_i/2)])
        q_vec = np.sin(theta_i/2) *  zone_vec_center
        zone_q = np.concatenate((q0,q_vec),0)
        zone_R = q2R(zone_q)
        zone_vec = zone_R @ zone_vec_0
        ax.plot([zone_vec[0]], [zone_vec[1]], [zone_vec[2]], 'y.')
    # zone1_vec[]

    # Plot the attitude history
    for t in range(T):
        if np.mod(t, 10) == 0:
            i = ib[:, t]
            j = jb[:, t]
            k = kb[:, t]
            ax.plot([i[0]], [i[1]], [i[2]], 'r.')
            i_line, = ax.plot([origin[0], i[0]], [origin[1], i[1]], [origin[2], i[2]], 'r-.')
            j_line, = ax.plot([origin[0], j[0]], [origin[1], j[1]], [origin[2], j[2]], 'g-.')
            k_line, = ax.plot([origin[0], k[0]], [origin[1], k[1]], [origin[2], k[2]], 'b-.')

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
            if t < T - 10:
                i_line.remove()
            j_line.remove()
            k_line.remove()
    # plt.clf()

    plt.show()
    return None


#########Global vriabls


Ts = 30  # 50 second
dt = 0.1
T = int(Ts / dt)  # Total time steps

# Assume symmetric cube
Ixx = 1
Iyy = 2
Izz = 2
J = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
n = 7
euler_des = np.array([0, 40, 40])
keep_out_att = np.array([0, 0, 40])
half_angle = 30

q_des = e2q(euler_des)
omega_des = np.zeros(3)
x_des = np.concatenate((omega_des, q_des), 0)
if __name__ == "__main__":

    q = np.zeros((4, T))
    omega = np.zeros((3, T))
    q[:, 0] = np.array([1, 0, 0, 0])
    x = np.concatenate((omega, q), 0)
    u = np.zeros((3, T - 1))
    u[0, :] = 0.001 * np.ones(T - 1)
    u[2, :] = 0.00 * np.ones(T - 1)
    u[1, :] = 0.00 * np.ones(T - 1)
    p = np.array([0, -0.001, 0.00, 0.00])
    ib = np.zeros((3, T))
    jb = np.zeros((3, T))
    kb = np.zeros((3, T))
    ib[:, 0] = np.array([1, 0, 0])
    jb[:, 0] = np.array([0, 1, 0])
    kb[:, 0] = np.array([0, 0, 1])
    x_traj = np.zeros([n, T])
    x_traj[3, :] = np.ones(T)
    u_traj = np.zeros([3, T - 1])
    # attitude_plot(euler_des, ib, jb, kb)
    [x_traj, u_traj] = tra_gen(x_traj, u_traj, x_des)
    for t in range(T - 1):
        q_t = x_traj[3:, t]
        R = q2R(q_t)
        ib[:, t] = ib[:, 0] @ R
        jb[:, t] = jb[:, 0] @ R
        kb[:, t] = kb[:, 0] @ R
    attitude_plot(euler_des, ib, jb, kb)

    time = np.linspace(0, Ts, T - 1)
    plt.subplot(3, 1, 1)
    plt.plot(time, u_traj[0, :])

    plt.subplot(3, 1, 2)
    plt.plot(time, u_traj[1, :])

    plt.subplot(3, 1, 3)
    plt.plot(time, u_traj[2, :])
    plt.show()
    # for t in range(T - 1):
    #     x_t = x[:, t]
    #     u_t = u[:, t]
    #     q_t = x_t[3:] / LA.norm(x_t[3:], 2)
    #     [A, B] = linearize(f_jax, x_t, u_t)
    #     A = np.asarray(A)
    #     B = np.asarray(B)
    #     # Continuous
    #     x_dot = f_continuous(x_t, u_t)
    #     # Linearized
    #     x_dot_ja = A @ x_t + B @ u_t
    #     # Discretized
    #     [Ad, Bd] = discretization(A, B)
    #     # x_tp1 = x_t + x_dot * dt # Continuous
    #     # x_tp1 = x_t + x_dot_ja * dt # Continuous and linearized
    #     x_tp1 = Ad @ x_t + Bd @ u_t  # Discretized
    #     x[:, t + 1] = x_tp1
    #     R = q2R(q_t)
    #
    #     ib[:, t] = ib[:, 0] @ R
    #     jb[:, t] = jb[:, 0] @ R
    #     kb[:, t] = kb[:, 0] @ R
    #     if np.mod(t, 40) == 0:
    #         print("x_dot", x_dot)
    #         print("x_dot_jax", x_dot_ja)
    #
    # attitude_plot(euler_des, ib, jb, kb)

    # t = np.linspace(0, 50, T)
    # plt.plot(t, euler[0] * 180 / np.pi, 'r.')
    # plt.plot(t, euler[1] * 180 / np.pi, 'g-')
    # plt.plot(t, euler[2] * 180 / np.pi, 'b.-')
    # plt.legend(['roll', 'pitch', 'yaw'])
    # plt.show()
