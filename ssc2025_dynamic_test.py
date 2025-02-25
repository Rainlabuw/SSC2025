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


#
def f_continuous_aero(
        x_t: np.ndarray,
        delta_t: np.ndarray
) -> np.ndarray:
    # Rotation matrix from body to LVLH
    R_t = q2R(x_t[3:])

    # Torque arm
    la = np.zeros((3, 10))
    la[:, 0] = np.array([-length, -1.5 * length, 0])
    la[:, 1] = np.array([-length, 1.5 * length, 0])
    la[:, 2] = np.array([-length, 0, 1.5 * length])
    la[:, 3] = np.array([-length, 0, -1.5 * length])

    la[:, 4] = np.array([length, 0, 0])
    la[:, 5] = np.array([-length, 0, 0])

    la[:, 6] = np.array([0, -width, 0])
    la[:, 7] = np.array([0, width, 0])

    la[:, 8] = np.array([0, 0, width])
    la[:, 9] = np.array([0, 0, -width])

    ############# Surface normals in wing and body frames
    surf_normal = np.zeros((3, 14))
    # For wings
    surf_normal[:, 0] = np.array([0, 0, 1])
    surf_normal[:, 1] = np.array([0, 0, -1])

    surf_normal[:, 2] = np.array([0, 0, 1])
    surf_normal[:, 3] = np.array([0, 0, -1])

    surf_normal[:, 4] = np.array([0, -1, 0])
    surf_normal[:, 5] = np.array([0, 1, 0])

    surf_normal[:, 6] = np.array([0, -1, 0])
    surf_normal[:, 7] = np.array([0, 1, 0])

    # For main body
    surf_normal[:, 8] = np.array([1, 0, 0])
    surf_normal[:, 9] = np.array([-1, 0, 0])

    surf_normal[:, 10] = np.array([0, -1, 0])
    surf_normal[:, 11] = np.array([0, 1, 0])

    surf_normal[:, 12] = np.array([0, 0, 1])
    surf_normal[:, 13] = np.array([0, 0, -1])

    ## Surface normals in wing and LVLH frame
    surf_normal_LVLH = np.zeros((3, 14))
    ## Surface normals of wings in body frame
    surf_normal_wing_body = np.zeros((3, 8))
    for i in range(8):
        wing_index = int(np.floor(i / 2))
        R_wing = R_nb(wing_index, delta_t)
        surf_normal_wing_body[:, i] = R_wing @ surf_normal[:, i]
        surf_normal_LVLH[:, i] = R_wing @ R_t @ surf_normal[:, i]
    for i in range(4):
        surf_normal_LVLH[:, i] = R_t @ surf_normal[:, i]

    ## Find u_t
    # Find the angle between the x-axis and the surface normals in the LVLH frame
    cosine_list = np.zeros(14)
    x_axis = np.array([1, 0, 0])
    for i in range(14):
        if i <= 7:
            x = np.dot(surf_normal_wing_body[:, i], x_axis)
        else:
            x = np.dot(surf_normal[:, i], x_axis)
        condlist = [x < 0, x >= 0]
        choicelist = [0, x]
        cosine_list[i] = np.select(condlist, choicelist)

    # Find the force acting on each surface
    dynamic_pressure = 0.1
    L_t = np.zeros((3, 14))  ## Lift acting on each surface
    areas = np.zeros(14)
    for i in range(8):
        areas[i] = 0.1 * 0.3
        L_t[:, i] = areas[i] * cosine_list[i] * dynamic_pressure * surf_normal_wing_body[:, i]

    # Total moment applied on the body
    u_t = np.zeros(3)

    for i in range(8):
        wing_index = int(np.floor(i / 2))
        u_t = u_t + np.cross(L_t[:, i], la[:, wing_index])

    # print(u_t)
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


def f_discretized_aero(
        x_t: np.ndarray,
        delta_t: np.ndarray
) -> np.ndarray:
    x_dot = f_continuous_aero(x_t, delta_t)
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


def f_jax_aero(
        x_t: jnp.ndarray,
        delta_t: jnp.ndarray
) -> np.ndarray:
    # Rotation matrix from body to LVLH
    R_t = q2R_jax(x_t[3:])

    width = 0.05
    length = 0.15
    # Torque arm
    la = jnp.zeros((3, 10))
    la = la.at[:, 0].set(jnp.array([-length, -1.5 * length, 0]))
    la = la.at[:, 1].set(jnp.array([-length, 1.5 * length, 0]))
    la = la.at[:, 2].set(jnp.array([-length, 0, 1.5 * length]))
    la = la.at[:, 3].set(jnp.array([-length, 0, -1.5 * length]))

    la = la.at[:, 4].set(jnp.array([length, 0, 0]))
    la = la.at[:, 5].set(jnp.array([-length, 0, 0]))

    la = la.at[:, 6].set(jnp.array([0, -width, 0]))
    la = la.at[:, 7].set(jnp.array([0, width, 0]))

    la = la.at[:, 8].set(jnp.array([0, 0, width]))
    la = la.at[:, 9].set(jnp.array([0, 0, -width]))

    ############# Surface normals in wing and body frames
    surf_normal = jnp.zeros((3, 14))
    # For wings
    surf_normal = surf_normal.at[:, 0].set(jnp.array([0, 0, 1]))
    surf_normal = surf_normal.at[:, 1].set(jnp.array([0, 0, -1]))

    surf_normal = surf_normal.at[:, 2].set(jnp.array([0, 0, 1]))
    surf_normal = surf_normal.at[:, 3].set(jnp.array([0, 0, -1]))

    surf_normal = surf_normal.at[:, 4].set(jnp.array([0, -1, 0]))
    surf_normal = surf_normal.at[:, 5].set(jnp.array([0, 1, 0]))

    surf_normal = surf_normal.at[:, 6].set(jnp.array([0, -1, 0]))
    surf_normal = surf_normal.at[:, 7].set(jnp.array([0, 1, 0]))

    # For main body
    surf_normal = surf_normal.at[:, 8].set(jnp.array([1, 0, 0]))
    surf_normal = surf_normal.at[:, 9].set(jnp.array([-1, 0, 0]))

    surf_normal = surf_normal.at[:, 10].set(jnp.array([0, -1, 0]))
    surf_normal = surf_normal.at[:, 11].set(jnp.array([0, 1, 0]))

    surf_normal = surf_normal.at[:, 12].set(jnp.array([0, 0, 1]))
    surf_normal = surf_normal.at[:, 13].set(jnp.array([0, 0, -1]))

    ## Surface normals in wing and LVLH frame
    surf_normal_LVLH = jnp.zeros((3, 14))
    ## Surface normals of wings in body frame
    surf_normal_wing_body = jnp.zeros((3, 8))

    for i in range(8):
        wing_index = int(np.floor(i / 2))

        if wing_index <= 1:
            R_wing = jnp.array([[jnp.cos(delta_t[wing_index]), 0, jnp.sin(delta_t[wing_index])],
                                [0, 1, 0],
                                [-jnp.sin(delta_t[wing_index]), 0, jnp.cos(delta_t[wing_index])]])
        else:
            R_wing = jnp.array([[jnp.cos(delta_t[wing_index]), -jnp.sin(delta_t[wing_index]), 0],
                                [jnp.sin(delta_t[wing_index]), jnp.cos(delta_t[wing_index]), 0],
                                [0, 0, 1]])

        # R_wing = R_nb_jax(wing_index, delta_t)
        # surf_normal_wing_body[:, i] = R_wing @ surf_normal[:, i]
        surf_normal_wing_body = surf_normal_wing_body.at[:, i].set(R_wing @ surf_normal[:, i])
        # surf_normal_LVLH[:, i] = R_wing @ R_t @ surf_normal[:, i]
        surf_normal_LVLH = surf_normal_LVLH.at[:, i].set(R_wing @ R_t @ surf_normal[:, i])
    for i in range(4):
        # surf_normal_LVLH[:, i] = R_t @ surf_normal[:, i]
        surf_normal_LVLH = surf_normal_LVLH.at[:, i].set(R_t @ surf_normal[:, i])

    ## Find u_t
    # Find the angle between the x-axis and the surface normals in the LVLH frame
    cosine_list = jnp.zeros(14)
    x_axis = jnp.array([1, 0, 0])
    for i in range(14):
        if i <= 7:
            x = jnp.dot(surf_normal_wing_body[:, i], x_axis)
        else:
            x = jnp.dot(surf_normal[:, i], x_axis)
        condlist = [x < 0, x >= 0]
        choicelist = [0, x]
        # cosine_list[i] = jnp.select(condlist, choicelist)
        cosine_list = cosine_list.at[i].set(jnp.select(condlist, choicelist))
    # Find the force acting on each surface
    dynamic_pressure = 0.1
    L_t = jnp.zeros((3, 14))  ## Lift acting on each surface
    areas = jnp.zeros(14)
    for i in range(8):
        # areas[i] = 0.1 * 0.3
        wing_index = int(np.floor(i / 2))
        areas = areas.at[i].set(0.1 * 0.3)
        # L_t[:, i] = areas[i] * cosine_list[i] * dynamic_pressure * surf_normal_wing_body[:, i]
        L_t = L_t.at[:, i].set(areas[i] * cosine_list[i] * dynamic_pressure * surf_normal_wing_body[:, i])

    # Total moment applied on the body
    h_t = jnp.zeros(3)

    for i in range(8):
        wing_index = int(np.floor(i / 2))
        h_t = h_t + jnp.cross(L_t[:, i], la[:, wing_index])
        # u_t = u_t + jnp.cross(la[:, wing_index], L_t[:, i])

    omega_t = x_t[0:3]
    q_t = x_t[3:]
    p = jnp.zeros(4)
    p = p.at[1:].set(omega_t)
    Omega = jnp.array([[p[0], -p[1], -p[2], -p[3]],
                       [p[1], p[0], p[3], -p[2]],
                       [p[2], -p[3], p[0], p[1]],
                       [p[3], p[2], -p[1], p[0]]])

    omegadot_t = jnp.linalg.inv(J) @ (h_t - jnp.cross(omega_t, (J @ omega_t)))
    qdot_t = 0.5 * Omega @ q_t
    # print(qdot_t)
    # qdot_t = np.zeros(4)

    x_dot_jax = jnp.concatenate((omegadot_t, qdot_t), 0)

    return x_dot_jax


def S_continuous(
        x_t: jnp.ndarray,
        u_t: jnp.ndarray
) -> jnp.ndarray:
    q = x_t[3:]
    R = jnp.array(
        [[1 - 2 * (q[2] * q[2] + q[3] * q[3]), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
         [2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] * q[1] + q[3] * q[3]), 2 * (q[2] * q[3] - q[0] * q[1])],
         [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])]])
    body_vec = R @ jnp.array([1, 0, 0])

    S_t = body_vec.T @ zone_vec_center - jnp.cos(half_angle * jnp.pi / 180)
    return S_t


def S_fun(
        x_t: np.ndarray,
) -> np.ndarray:
    q = x_t[3:]
    R = np.array(
        [[1 - 2 * (q[2] * q[2] + q[3] * q[3]), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
         [2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] * q[1] + q[3] * q[3]), 2 * (q[2] * q[3] - q[0] * q[1])],
         [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])]])
    body_vec = R @ np.array([1, 0, 0])

    S = body_vec.T @ zone_vec_center - np.cos(half_angle * np.pi / 180)
    return S


def S_linearize(
        S_continuous: jnp.ndarray,
        x_t: np.ndarray,
        u_t: np.ndarray
):
    # Compute the Jacobian of f(x, u) with respect to x (A matrix)
    # A = jax.jacobian(lambda x: f_jax(x, u_t))(x_t)
    dSdx = jax.jacfwd(lambda x: S_continuous(x, u_t))(x_t)
    # Compute the Jacobian of f(x, u) with respect to u (B matrix)
    dSdu = jax.jacfwd(lambda u: S_continuous(x_t, u))(u_t)
    return dSdx, dSdu


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


def linearize_aero(
        f_jax_aero: jnp.ndarray,
        x_t: np.ndarray,
        delta_t: np.ndarray
):
    # Compute the Jacobian of f(x, u) with respect to x (A matrix)
    # A = jax.jacobian(lambda x: f_jax(x, u_t))(x_t)
    A = jax.jacobian(lambda x: f_jax_aero(x, delta_t))(x_t)
    # Compute the Jacobian of f(x, u) with respect to u (B matrix)
    # B = jax.jacobian(lambda u: f_jax(x_t, u))(u_t)
    B = jax.jacobian(lambda delta: f_jax_aero(x_t, delta))(delta_t)

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


def discretization_aero(
        A: np.ndarray,
        B: np.ndarray
) -> list:
    C = np.eye(7)
    D = np.zeros((7, 4))
    sys = signal.StateSpace(A, B, C, D)
    sysd = sys.to_discrete(dt)
    Ad = sysd.A
    Bd = sysd.B

    return Ad, Bd


def sub_problem_cost_fun(
        lambda_param: float,
        w: cp.Variable,
        v: cp.Variable,
        s: cp.Variable,
        U_traj: np.ndarray
):
    sub_problem_cost = 0 * lambda_param * cp.norm(((U_traj + w)), 1) + 10 * lambda_param * cp.sum(
        cp.sum(cp.abs(v))) + 1 * lambda_param * cp.sum(cp.pos(s))
    return sub_problem_cost


def solve_convex_optimal_control_subproblem(
        X_traj: np.ndarray,
        U_traj: np.ndarray,
        dSdx: np.ndarray,
        x_des: np.ndarray,
        r: np.ndarray,
        i: int
) -> list:
    # r = 0.2
    lambda_param = 10000
    # Define variables for optimization
    w = cp.Variable((3, T - 1))
    v = cp.Variable((n, T - 1))
    d = cp.Variable((n, T))
    s = cp.Variable(T)
    constraints = [d[:, 0] == np.zeros(7)]
    E = np.eye(7)
    sup_problem_cost = sub_problem_cost_fun(lambda_param, w, v, s, U_traj)
    S = np.zeros(T)
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
        # [dSdx, dSdu] = S_linearize(S_continuous, x_t, u_t)

        # Dynamic constraints
        constraints.append(
            x_tp1 + d_tp1 ==
            f_discretized(x_t, u_t) +
            Ad @ d_t + Bd @ w_t + 1 * E @ v_t)
        constraints.append(cp.abs(w_t) <= r)
        if i <= 0:
            constraints.append(
                x_tp1 + d_tp1 ==
                f_discretized(x_t, u_t) +
                Ad @ d_t + Bd @ w_t + 1 * E @ v_t)
            constraints.append(cp.abs(w_t) <= r)
        else:
            constraints.append(
                x_tp1 + d_tp1 ==
                f_discretized(x_t, u_t) +
                Ad @ d_t + Bd @ w_t + 0 * E @ v_t)
            constraints.append(cp.abs(w_t) <= r)
            # Keep out zone constraints
            S_t = S_fun(x_t)
            S[t] = S_t
            dSdx_t = np.asarray(dSdx)
            # dSdu = np.asarray(dSdu)
            dSdx_t = dSdx[:, t]
            constraints.append(S_t + dSdx_t @ d_t <= s[t])
            constraints.append(s[t] >= 0)
        # constraints.append(cp.abs(u_t + w_t) <= 0.02)
        # # Keep out zone constraints
        # S_t = S_fun(x_t)
        # S[t] = S_t
        # dSdx = np.asarray(dSdx)
        # dSdu = np.asarray(dSdu)
        # constraints.append(S_t + dSdx @ d_t <= s[t])
        # constraints.append(s[t] >= 0)

    # Terminal condition
    constraints.append(X_traj[:, T - 1] + d[:, T - 1] == np.array(
        [x_des[0], x_des[1], x_des[2], x_des[3], x_des[4], x_des[5], x_des[6]]))

    # Define the problem
    problem = cp.Problem(cp.Minimize(sup_problem_cost), constraints)
    problem.solve(solver=cp.CLARABEL)
    cost = problem.value
    w_traj_val = w.value
    d_traj_val = d.value
    return cost, d_traj_val, w_traj_val


def solve_convex_optimal_control_subproblem_aero(
        X_traj: np.ndarray,
        delta_traj: np.ndarray,
        dSdx: np.ndarray,
        x_des: np.ndarray,
        r: np.ndarray,
        i: int
) -> list:

    lambda_param = 10000
    # Define variables for optimization
    w = cp.Variable((4, T - 1))
    v = cp.Variable((n, T - 1))
    d = cp.Variable((n, T))
    s = cp.Variable(T)
    constraints = [d[:, 0] == np.zeros(7)]
    E = np.eye(7)
    sup_problem_cost = sub_problem_cost_fun(lambda_param, w, v, s, delta_traj)
    S = np.zeros(T)
    for t in range(T - 1):
        x_t = X_traj[:, t]
        x_tp1 = X_traj[:, t + 1]
        d_t = d[:, t]
        d_tp1 = d[:, t + 1]
        delta_t = delta_traj[:, t]
        w_t = w[:, t]
        v_t = v[:, t]
        [A, B] = linearize_aero(f_jax_aero, x_t, delta_t)
        [Ad, Bd] = discretization_aero(A, B)
        # [dSdx, dSdu] = S_linearize(S_continuous, x_t, u_t)

        # Dynamic constraints
        constraints.append(
            x_tp1 + d_tp1 ==
            f_discretized_aero(x_t, delta_t) +
            Ad @ d_t + Bd @ w_t + 1 * E @ v_t)
        constraints.append(cp.abs(w_t) <= r)
        for wing in range(4):
            constraints.append(cp.abs(delta_t[wing] + w_t[wing]) <= 3.1415 / 4.2)
        if i <= 1:
            constraints.append(
                x_tp1 + d_tp1 ==
                f_discretized_aero(x_t, delta_t) +
                Ad @ d_t + Bd @ w_t + 1 * E @ v_t)
            constraints.append(cp.abs(w_t) <= r)
        else:
            constraints.append(
                x_tp1 + d_tp1 ==
                f_discretized_aero(x_t, delta_t) +
                Ad @ d_t + Bd @ w_t + 1 * E @ v_t)
            constraints.append(cp.abs(w_t) <= r)
            # Keep out zone constraints
            S_t = S_fun(x_t)
            S[t] = S_t
            dSdx_t = np.asarray(dSdx)
            # dSdu = np.asarray(dSdu)
            dSdx_t = dSdx[:, t]
            constraints.append(S_t + dSdx_t @ d_t <= s[t])
            constraints.append(s[t] >= 0)

    # Terminal condition
    # constraints.append(X_traj[:, T - 1] + d[:, T - 1] == np.array(
    #     [x_des[0], x_des[1], x_des[2], x_des[3], x_des[4], x_des[5], x_des[6]]))
    constraints.append(X_traj[3:, T - 1] + d[3:, T - 1] == np.array([x_des[3], x_des[4], x_des[5], x_des[6]]))
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
    # iter = 0
    iter = 11
    cost_list = np.zeros(iter)
    r = 1
    dSdx = np.zeros((n, T))
    for i in range(iter):

        # for linearization
        for t in range(T - 1):
            x_t = X_traj[:, t]
            u_t = U_traj[:, t]
            [dSdx_t, dSdu_t] = S_linearize(S_continuous, x_t, u_t)
            dSdx[:, t] = np.asarray(dSdx_t)

        [cost, d_traj_val, w_traj_val] = solve_convex_optimal_control_subproblem(X_traj, U_traj, dSdx, x_des, r, i)
        X_traj = X_traj + d_traj_val
        U_traj = U_traj + w_traj_val

        print('Iteration:   ', i, '    Cost:   ', cost, '   Trust region:  ', r)
        cost_list[i] = cost
        r = trust_region_update(cost_list, i, r)

    return X_traj, U_traj


def tra_gen_aero(
        X_traj: np.ndarray,
        delta_traj: np.ndarray,
        x_des: np.ndarray
) -> list:
    # iter = 0
    iter = 7
    cost_list = np.zeros(iter)
    r = 2
    dSdx = np.zeros((n, T))
    for i in range(iter):

        # for linearization
        for t in range(T - 1):
            x_t = X_traj[:, t]
            delta_t = delta_traj[:, t]
            [dSdx_t, dSdu_t] = S_linearize(S_continuous, x_t, delta_t)
            dSdx[:, t] = np.asarray(dSdx_t)

        [cost, d_traj_val, w_traj_val] = solve_convex_optimal_control_subproblem_aero(X_traj, delta_traj, dSdx, x_des,
                                                                                      r, i)
        X_traj = X_traj + d_traj_val
        delta_traj = delta_traj + w_traj_val

        print('Iteration:   ', i, '    Cost:   ', cost, '   Trust region:  ', r)
        cost_list[i] = cost
        r = trust_region_update(cost_list, i, r)

    return X_traj, delta_traj


def trust_region_update(
        cost_list: np.ndarray,
        iter: int,
        r_current: np.ndarray
) -> np.ndarray:
    rho0 = 0.1
    rho1 = 0.25
    rho2 = 0.7
    r_default = 1.5
    if iter >= 1:
        delta_L = (cost_list[iter] - cost_list[iter - 1]) / cost_list[iter]
    else:
        delta_L = 1

    if cost_list[iter] <= 500:
        if np.abs(delta_L) <= rho0:
            r_next = np.max((r_current / 2, 0.002))
        elif np.abs(delta_L) <= rho1:
            r_next = np.max((r_current / 1.2, 0.02))
        elif np.abs(delta_L) <= rho2:
            r_next = np.max((r_current / 2.2, 0.02))
        else:
            r_next = np.max((r_current / 1.1, 0.02))
    else:
        r_next = r_default

    return r_next


# Rotation matrix from the body coord to the wing coord
def R_nb(
        index: int,  # index of the wing
        delta: np.ndarray
) -> np.ndarray:
    if index <= 1:
        R_y = np.array([[np.cos(delta[index]), 0, np.sin(delta[index])],
                        [0, 1, 0],
                        [-np.sin(delta[index]), 0, np.cos(delta[index])]])
        R = R_y
    else:
        R_z = np.array([[np.cos(delta[index]), -np.sin(delta[index]), 0],
                        [np.sin(delta[index]), np.cos(delta[index]), 0],
                        [0, 0, 1]])
        R = R_z
    return R


def R_nb_jax(
        index: int,  # index of the wing
        delta: jnp.ndarray
) -> jnp.ndarray:
    if index <= 1:
        R_y = jnp.array([[jnp.cos(delta[index]), 0, jnp.sin(delta[index])],
                         [0, 1, 0],
                         [-jnp.sin(delta[index]), 0, jnp.cos(delta[index])]])
        R = R_y
    else:
        R_z = jnp.array([[jnp.cos(delta[index]), -jnp.sin(delta[index]), 0],
                         [jnp.sin(delta[index]), jnp.cos(delta[index]), 0],
                         [0, 0, 1]])
        R = R_z
    return R


# From quaternion to rotation matrix from body frame to LVLH (123)
def q2R(
        q: np.ndarray
) -> np.ndarray:
    R = np.array(
        [[1 - 2 * (q[2] * q[2] + q[3] * q[3]), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
         [2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] * q[1] + q[3] * q[3]), 2 * (q[2] * q[3] - q[0] * q[1])],
         [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])]])
    return R


def q2R_jax(
        q: jnp.ndarray
) -> jnp.ndarray:
    R = jnp.array(
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
        x_traj: np.ndarray,
        euler_des: np.ndarray,
        ib: np.ndarray,
        jb: np.ndarray,
        kb: np.ndarray,
        delta: np.ndarray
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
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot([zone_vec_center[0]], [zone_vec_center[1]], [zone_vec_center[2]], 'co')
    for i in range(200):
        theta_i = theta[i]
        q0 = np.array([np.cos(theta_i / 2)])
        q_vec = np.sin(theta_i / 2) * zone_vec_center
        zone_q = np.concatenate((q0, q_vec), 0)
        zone_R = q2R(zone_q)
        zone_vec = zone_R @ zone_vec_0
        ax.plot([zone_vec[0]], [zone_vec[1]], [zone_vec[2]], 'y.')
    # zone1_vec[]

    # Plot the attitude and wing rotation history
    # Satellite animation dimension
    w = 0.2
    l = 0.5
    nodes = np.zeros((8, 3, T))
    axes = np.zeros((3, 8))
    axes_t = np.zeros((3, 8))
    # For satellite body
    axes[:, 0] = np.array([l, -w, w])
    axes[:, 1] = np.array([l, w, w])
    axes[:, 2] = np.array([l, w, -w])
    axes[:, 3] = np.array([l, -w, -w])
    axes[:, 4] = np.array([-l, -w, w])
    axes[:, 5] = np.array([-l, w, w])
    axes[:, 6] = np.array([-l, w, -w])
    axes[:, 7] = np.array([-l, -w, -w])
    # For the wing
    wing_center = np.zeros((3, 8))
    wing_center[:, 0] = np.array([-l, -2 * l - 0.5 * w, 0])
    wing_center[:, 1] = np.array([-l, -0.5 * w, 0])
    wing_center[:, 2] = np.array([-l, 2 * l + 0.5 * w, 0])
    wing_center[:, 3] = np.array([-l, 0.5 * w, 0])
    wing_center[:, 4] = np.array([-l, 0, 2 * l + 0.5 * w])
    wing_center[:, 5] = np.array([-l, 0, +0.5 * w])
    wing_center[:, 6] = np.array([-l, 0, -2 * l - 0.5 * w])
    wing_center[:, 7] = np.array([-l, 0, -0.5 * w])
    wing_all_node = np.zeros((3, 16))
    # wing 1
    # wing_all_node[:, 0] = np.array([-l + 0.5 * w, -2 * l, 0])
    # wing_all_node[:, 1] = np.array([-l - 0.5 * w, -2 * l, 0])
    # wing_all_node[:, 2] = np.array([-l + 0.5 * w, -w, 0])
    # wing_all_node[:, 3] = np.array([-l - 0.5 * w, -w, 0])
    wing_all_node[:, 0] = np.array([0.5 * w, 0, 0])
    wing_all_node[:, 1] = np.array([-0.5 * w, 0, 0])
    wing_all_node[:, 2] = np.array([0.5 * w, 0, 0])
    wing_all_node[:, 3] = np.array([- 0.5 * w, 0, 0])
    # wing 2
    wing_all_node[:, 4] = np.array([0.5 * w, 0, 0])
    wing_all_node[:, 5] = np.array([- 0.5 * w, 0, 0])
    wing_all_node[:, 6] = np.array([0.5 * w, 0, 0])
    wing_all_node[:, 7] = np.array([- 0.5 * w, 0, 0])
    # wing 3
    wing_all_node[:, 8] = np.array([0.5 * w, 0, 0])
    wing_all_node[:, 9] = np.array([- 0.5 * w, 0, 0])
    wing_all_node[:, 10] = np.array([0.5 * w, 0, 0])
    wing_all_node[:, 11] = np.array([- 0.5 * w, 0, 0])
    # wing 4
    wing_all_node[:, 12] = np.array([0.5 * w, 0, 0])
    wing_all_node[:, 13] = np.array([- 0.5 * w, 0, 0])
    wing_all_node[:, 14] = np.array([0.5 * w, 0, 0])
    wing_all_node[:, 15] = np.array([- 0.5 * w, 0, 0])
    wing_all_node_t = np.zeros((3, 16))  # this is the reference node for wing with no rotation
    wing_center_t = np.zeros((3, 8))
    #################### Plotting animation
    for t in range(T-1):
        delta_t = delta[:, t]
        R_t = q2R(x_traj[3:, t])
        if np.mod(t, 1) == 0:
            for j in range(8):
                axes_t[:, j] = R_t @ axes[:, j]

            # for k in range(16):
            #     index = int(np.floor(k / 4))
            #     R_wing = R_nb(index, delta_t)
            #     wing_all_node_t[:, k] = R_wing @ R_t @ wing_all_node[:, k]
            for center_count in range(8):
                wing_center_t[:, center_count] = R_t @ wing_center[:, center_count]
                for wing_node in range(2):
                    node_index = center_count * 2 + wing_node
                    wing_index = int(np.floor(center_count / 2))
                    R_node = R_nb(wing_index, delta_t)
                    wing_all_node_t[:, node_index] = R_t @ R_node @ wing_all_node[:, node_index] + wing_center_t[:,
                                                                                                   center_count]

            ## Plotting wings
            # Wing 1
            edge_wing_1_edge1, = ax.plot([wing_all_node_t[0, 1], wing_all_node_t[0, 0]],
                                         [wing_all_node_t[1, 1], wing_all_node_t[1, 0]],
                                         [wing_all_node_t[2, 1], wing_all_node_t[2, 0]],
                                         'g')
            edge_wing_1_edge2, = ax.plot([wing_all_node_t[0, 1], wing_all_node_t[0, 3]],
                                         [wing_all_node_t[1, 1], wing_all_node_t[1, 3]],
                                         [wing_all_node_t[2, 1], wing_all_node_t[2, 3]],
                                         'g')
            edge_wing_1_edge3, = ax.plot([wing_all_node_t[0, 3], wing_all_node_t[0, 2]],
                                         [wing_all_node_t[1, 3], wing_all_node_t[1, 2]],
                                         [wing_all_node_t[2, 3], wing_all_node_t[2, 2]],
                                         'g')
            edge_wing_1_edge4, = ax.plot([wing_all_node_t[0, 2], wing_all_node_t[0, 0]],
                                         [wing_all_node_t[1, 2], wing_all_node_t[1, 0]],
                                         [wing_all_node_t[2, 2], wing_all_node_t[2, 0]],
                                         'g')
            # # # Wing 2
            edge_wing_2_edge1, = ax.plot([wing_all_node_t[0, 1 + 4], wing_all_node_t[0, 0 + 4]],
                                         [wing_all_node_t[1, 1 + 4], wing_all_node_t[1, 0 + 4]],
                                         [wing_all_node_t[2, 1 + 4], wing_all_node_t[2, 0 + 4]],
                                         'g')
            edge_wing_2_edge2, = ax.plot([wing_all_node_t[0, 1 + 4], wing_all_node_t[0, 3 + 4]],
                                         [wing_all_node_t[1, 1 + 4], wing_all_node_t[1, 3 + 4]],
                                         [wing_all_node_t[2, 1 + 4], wing_all_node_t[2, 3 + 4]],
                                         'g')
            edge_wing_2_edge3, = ax.plot([wing_all_node_t[0, 3 + 4], wing_all_node_t[0, 2 + 4]],
                                         [wing_all_node_t[1, 3 + 4], wing_all_node_t[1, 2 + 4]],
                                         [wing_all_node_t[2, 3 + 4], wing_all_node_t[2, 2 + 4]],
                                         'g')
            edge_wing_2_edge4, = ax.plot([wing_all_node_t[0, 2 + 4], wing_all_node_t[0, 0 + 4]],
                                         [wing_all_node_t[1, 2 + 4], wing_all_node_t[1, 0 + 4]],
                                         [wing_all_node_t[2, 2 + 4], wing_all_node_t[2, 0 + 4]],
                                         'g')

            # Wing 2
            edge_wing_3_edge1, = ax.plot([wing_all_node_t[0, 1 + 4 + 4], wing_all_node_t[0, 0 + 4 + 4]],
                                         [wing_all_node_t[1, 1 + 4 + 4], wing_all_node_t[1, 0 + 4 + 4]],
                                         [wing_all_node_t[2, 1 + 4 + 4], wing_all_node_t[2, 0 + 4 + 4]],
                                         'g')
            edge_wing_3_edge2, = ax.plot([wing_all_node_t[0, 1 + 4 + 4], wing_all_node_t[0, 3 + 4 + 4]],
                                         [wing_all_node_t[1, 1 + 4 + 4], wing_all_node_t[1, 3 + 4 + 4]],
                                         [wing_all_node_t[2, 1 + 4 + 4], wing_all_node_t[2, 3 + 4 + 4]],
                                         'g')
            edge_wing_3_edge3, = ax.plot([wing_all_node_t[0, 3 + 4 + 4], wing_all_node_t[0, 2 + 4 + 4]],
                                         [wing_all_node_t[1, 3 + 4 + 4], wing_all_node_t[1, 2 + 4 + 4]],
                                         [wing_all_node_t[2, 3 + 4 + 4], wing_all_node_t[2, 2 + 4 + 4]],
                                         'g')
            edge_wing_3_edge4, = ax.plot([wing_all_node_t[0, 2 + 4 + 4], wing_all_node_t[0, 0 + 4 + 4]],
                                         [wing_all_node_t[1, 2 + 4 + 4], wing_all_node_t[1, 0 + 4 + 4]],
                                         [wing_all_node_t[2, 2 + 4 + 4], wing_all_node_t[2, 0 + 4 + 4]],
                                         'g')
            # Wing 2
            edge_wing_4_edge1, = ax.plot([wing_all_node_t[0, 1 + 4 + 4 + 4], wing_all_node_t[0, 0 + 4 + 4 + 4]],
                                         [wing_all_node_t[1, 1 + 4 + 4 + 4], wing_all_node_t[1, 0 + 4 + 4 + 4]],
                                         [wing_all_node_t[2, 1 + 4 + 4 + 4], wing_all_node_t[2, 0 + 4 + 4 + 4]],
                                         'g')
            edge_wing_4_edge2, = ax.plot([wing_all_node_t[0, 1 + 4 + 4 + 4], wing_all_node_t[0, 3 + 4 + 4 + 4]],
                                         [wing_all_node_t[1, 1 + 4 + 4 + 4], wing_all_node_t[1, 3 + 4 + 4 + 4]],
                                         [wing_all_node_t[2, 1 + 4 + 4 + 4], wing_all_node_t[2, 3 + 4 + 4 + 4]],
                                         'g')
            edge_wing_4_edge3, = ax.plot([wing_all_node_t[0, 3 + 4 + 4 + 4], wing_all_node_t[0, 2 + 4 + 4 + 4]],
                                         [wing_all_node_t[1, 3 + 4 + 4 + 4], wing_all_node_t[1, 2 + 4 + 4 + 4]],
                                         [wing_all_node_t[2, 3 + 4 + 4 + 4], wing_all_node_t[2, 2 + 4 + 4 + 4]],
                                         'g')
            edge_wing_4_edge4, = ax.plot([wing_all_node_t[0, 2 + 4 + 4 + 4], wing_all_node_t[0, 0 + 4 + 4 + 4]],
                                         [wing_all_node_t[1, 2 + 4 + 4 + 4], wing_all_node_t[1, 0 + 4 + 4 + 4]],
                                         [wing_all_node_t[2, 2 + 4 + 4 + 4], wing_all_node_t[2, 0 + 4 + 4 + 4]],
                                         'g')
            ## Satellite body edges
            edge_1, = ax.plot([axes_t[0, 1], axes_t[0, 0]], [axes_t[1, 1], axes_t[1, 0]], [axes_t[2, 1], axes_t[2, 0]],
                              'b')
            edge_2, = ax.plot([axes_t[0, 2], axes_t[0, 1]], [axes_t[1, 2], axes_t[1, 1]], [axes_t[2, 2], axes_t[2, 1]],
                              'b')
            edge_3, = ax.plot([axes_t[0, 3], axes_t[0, 2]], [axes_t[1, 3], axes_t[1, 2]], [axes_t[2, 3], axes_t[2, 2]],
                              'b')
            edge_4, = ax.plot([axes_t[0, 0], axes_t[0, 3]], [axes_t[1, 0], axes_t[1, 3]], [axes_t[2, 0], axes_t[2, 3]],
                              'b')
            edge_5, = ax.plot([axes_t[0, 5], axes_t[0, 4]], [axes_t[1, 5], axes_t[1, 4]], [axes_t[2, 5], axes_t[2, 4]],
                              'b')
            edge_6, = ax.plot([axes_t[0, 6], axes_t[0, 5]], [axes_t[1, 6], axes_t[1, 5]], [axes_t[2, 6], axes_t[2, 5]],
                              'b')
            edge_7, = ax.plot([axes_t[0, 7], axes_t[0, 6]], [axes_t[1, 7], axes_t[1, 6]], [axes_t[2, 7], axes_t[2, 6]],
                              'b')
            edge_8, = ax.plot([axes_t[0, 4], axes_t[0, 7]], [axes_t[1, 4], axes_t[1, 7]], [axes_t[2, 4], axes_t[2, 7]],
                              'b')
            edge_9, = ax.plot([axes_t[0, 0], axes_t[0, 4]], [axes_t[1, 0], axes_t[1, 4]], [axes_t[2, 0], axes_t[2, 4]],
                              'b')
            edge_10, = ax.plot([axes_t[0, 1], axes_t[0, 5]], [axes_t[1, 1], axes_t[1, 5]], [axes_t[2, 1], axes_t[2, 5]],
                               'b')
            edge_11, = ax.plot([axes_t[0, 2], axes_t[0, 6]], [axes_t[1, 2], axes_t[1, 6]], [axes_t[2, 2], axes_t[2, 6]],
                               'b')
            edge_12, = ax.plot([axes_t[0, 3], axes_t[0, 7]], [axes_t[1, 3], axes_t[1, 7]], [axes_t[2, 3], axes_t[2, 7]],
                               'b')

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
            plt.pause(0.13)
            if t < T - 1:
                ## Remove excessive lines
                # Remove wing edges
                edge_wing_1_edge1.remove()
                edge_wing_1_edge2.remove()
                edge_wing_1_edge3.remove()
                edge_wing_1_edge4.remove()
                edge_wing_2_edge1.remove()
                edge_wing_2_edge2.remove()
                edge_wing_2_edge3.remove()
                edge_wing_2_edge4.remove()
                edge_wing_3_edge1.remove()
                edge_wing_3_edge2.remove()
                edge_wing_3_edge3.remove()
                edge_wing_3_edge4.remove()
                edge_wing_4_edge1.remove()
                edge_wing_4_edge2.remove()
                edge_wing_4_edge3.remove()
                edge_wing_4_edge4.remove()
                # Remove body edges
                i_line.remove()
                edge_1.remove()
                edge_2.remove()
                edge_3.remove()
                edge_4.remove()
                edge_5.remove()
                edge_6.remove()
                edge_7.remove()
                edge_8.remove()
                edge_9.remove()
                edge_10.remove()
                edge_11.remove()
                edge_12.remove()

            j_line.remove()
            k_line.remove()
    # plt.clf()

    plt.show()
    return None


#########Global vriabls


Ts = 120  # 50 second
dt = 2
T = int(Ts / dt)  # Total time steps

# Assume symmetric cube
Ixx = 1
Iyy = 3
Izz = 3
J = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
n = 7
euler_des = np.array([20, -35, 0])
# euler_des = np.array([0.5, 0, 0.5])
keep_out_att = np.array([0, -15.5, 0])
half_angle = 12
x_axis = np.array([1, 0, 0])
zone_att_0 = np.array([0, keep_out_att[1] + half_angle, keep_out_att[2]])
zone_q_0 = e2q(zone_att_0)
zone_R_0 = q2R(zone_q_0)
zone_vec_0 = zone_R_0 @ x_axis
zone_q_center = e2q(keep_out_att)
zone_vec_center = q2R(zone_q_center) @ x_axis

# Geometry of satellite
width = 0.05
length = 0.15

q_des = e2q(euler_des)
omega_des = np.zeros(3)
x_des = np.concatenate((omega_des, q_des), 0)
if __name__ == "__main__":
    ################################# Active aerodynamic control case
    q = np.zeros((4, T))
    omega = np.zeros((3, T))
    q[:, 0] = np.array([1, 0, 0, 0])
    delta = np.zeros((4, T))  # Deflection angle for wings
    delta[0, :] = np.pi / 5
    delta[1, :] = -np.pi / 5
    delta[2, :] = np.pi / 5
    delta[3, :] = -np.pi / 5
    x = np.concatenate((omega, q), 0)
    ib = np.zeros((3, T))
    jb = np.zeros((3, T))
    kb = np.zeros((3, T))
    ib[:, 0] = np.array([1, 0, 0])
    jb[:, 0] = np.array([0, 1, 0])
    kb[:, 0] = np.array([0, 0, 1])
    x_traj = np.zeros([n, T])
    x_traj[3, :] = np.ones(T)
    delta_traj = np.zeros([4, T - 1])

    # attitude_plot(x_traj, euler_des, ib, jb, kb,delta)
    [x_traj, delta_traj] = tra_gen_aero(x_traj, delta_traj, x_des)
    S = np.zeros(T)
    for t in range(T - 1):
        q_t = x_traj[3:, t]
        R = q2R(q_t)
        ib[:, t] = R @ ib[:, 0]
        jb[:, t] = R @ jb[:, 0]
        kb[:, t] = R @ kb[:, 0]

        body_vec = q2R(x_traj[3:, t]) @ np.array([1, 0, 0])
        # S_t = body_vec.T @ zone_vec_center - np.cos(half_angle*np.pi/180)
        S_t = body_vec.T @ zone_vec_center
        S[t] = S_t


    time = np.linspace(0, Ts, T - 1)
    plt.subplot(4, 1, 1)
    plt.plot(time, delta_traj[0, :] * 180 / np.pi)
    plt.xlabel("Time")
    plt.ylabel("Angle")
    plt.subplot(4, 1, 2)
    plt.plot(time, delta_traj[1, :] * 180 / np.pi)
    plt.xlabel("Time")
    plt.ylabel("Angle")
    plt.subplot(4, 1, 3)
    plt.plot(time, delta_traj[2, :] * 180 / np.pi)
    plt.xlabel("Time")
    plt.ylabel("Angle")
    plt.subplot(4, 1, 4)
    plt.plot(time, delta_traj[3, :] * 180 / np.pi)
    plt.xlabel("Time")
    plt.ylabel("Angle")
    plt.show()
    attitude_plot(x_traj, euler_des, ib, jb, kb, delta_traj)
    ################################ Reaction wheel case
    q = np.zeros((4, T))
    omega = np.zeros((3, T))
    q[:, 0] = np.array([1, 0, 0, 0])
    delta = np.zeros((4, T))  # Deflection angle for wings
    # delta[0, :] = np.pi / 4
    # delta[1, :] = -np.pi / 4
    # delta[2, :] = np.pi / 4
    # delta[3, :] = -np.pi / 3
    x = np.concatenate((omega, q), 0)
    ib = np.zeros((3, T))
    jb = np.zeros((3, T))
    kb = np.zeros((3, T))
    ib[:, 0] = np.array([1, 0, 0])
    jb[:, 0] = np.array([0, 1, 0])
    kb[:, 0] = np.array([0, 0, 1])
    x_traj = np.zeros([n, T])
    x_traj[3, :] = np.ones(T)
    delta_traj = np.zeros([3, T - 1])
    u_traj = np.zeros([3, T - 1])
    # attitude_plot(x_traj, euler_des, ib, jb, kb,delta)
    [x_traj, u_traj] = tra_gen(x_traj, u_traj, x_des)
    S = np.zeros(T)
    for t in range(T - 1):
        q_t = x_traj[3:, t]
        R = q2R(q_t)
        ib[:, t] = R @ ib[:, 0]
        jb[:, t] = R @ jb[:, 0]
        kb[:, t] = R @ kb[:, 0]

        body_vec = q2R(x_traj[3:, t]) @ np.array([1, 0, 0])
        # S_t = body_vec.T @ zone_vec_center - np.cos(half_angle*np.pi/180)
        S_t = body_vec.T @ zone_vec_center
        S[t] = S_t
    attitude_plot(x_traj, euler_des, ib, jb, kb, u_traj)

    time = np.linspace(0, Ts, T - 1)
    plt.subplot(3, 1, 1)
    plt.plot(time, u_traj[0, :])

    plt.subplot(3, 1, 2)
    plt.plot(time, u_traj[1, :])

    plt.subplot(3, 1, 3)
    plt.plot(time, u_traj[2, :])
    plt.show()

    ######################### Dynamic test(No control)

    q = np.zeros((4, T))
    omega = np.zeros((3, T))
    q[:, 0] = np.array([1, 0, 0, 0])
    delta = np.zeros((4, T))  # Deflection angle for wings
    #### Valid linearization range is from -pi/4.5 to pi/4.5
    delta[0, :] = np.pi / 4.5
    delta[1, :] = np.pi / 4.5
    delta[2, :] = np.pi / 4.5
    # delta[3, :] = -np.pi / 4.5
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

    for t in range(T - 1):
        print("Progress:    ", t / T * 100)
        delta_t = delta[:, t]
        x_t = x_traj[:, t]
        u_t = u[:, t]
        q_t = x_t[3:] / LA.norm(x_t[3:], 2)
        [A, B] = linearize_aero(f_jax_aero, x_t, delta_t)
        A = np.asarray(A)
        B = np.asarray(B)
        # Continuous
        # x_dot = f_continuous(x_t, u_t)
        # x_dot = f_continuous_aero(x_t, delta_t)
        x_dot = A @ x_t + B @ delta_t
        # Linearized
        # x_dot_ja = A @ x_t + B @ u_t
        # Discretized
        # [Ad, Bd] = discretization(A, B)
        x_tp1 = x_t + x_dot * dt  # Continuous
        # x_tp1 = x_t + x_dot_ja * dt # Continuous and linearized
        # x_tp1 = Ad @ x_t + Bd @ u_t  # Discretized
        x_traj[:, t + 1] = x_tp1
        R = q2R(q_t)

        ib[:, t] = R @ ib[:, 0]
        jb[:, t] = R @ jb[:, 0]
        kb[:, t] = R @ kb[:, 0]
        # if np.mod(t, 40) == 0:
        #     print("x_dot", x_dot)
        #     print("x_dot_jax", x_dot_ja)

    attitude_plot(x_traj, euler_des, ib, jb, kb, delta)

    # t = np.linspace(0, 50, T)
    # plt.plot(t, euler[0] * 180 / np.pi, 'r.')
    # plt.plot(t, euler[1] * 180 / np.pi, 'g-')
    # plt.plot(t, euler[2] * 180 / np.pi, 'b.-')
    # plt.legend(['roll', 'pitch', 'yaw'])
    # plt.show()
