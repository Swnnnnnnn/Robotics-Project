import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import ZMP
from steps_generator import CandidateFootstepGenerator


# ==========================================================
# 1. SIMULATION AND ROBOT PARAMETERS
# ==========================================================

# MPC parameters
N_horizon = 30 * 2
T_step = 0.05 / 2
g = 9.81

# Robot physical parameters
h_com = 0.85
l = 0.35
foot_size_lat = 0.04
foot_size_fwd = 0.07
m_robot = 75.0
v_max = 0.5

# Commanded motion
v_command = [0.0, 0.3, 0.0]
total_duration = 20.0
start_pose = [0.0, -0.1, 0.0]

# External disturbance parameters
F_lat = 0.0
F_fwd = 320.0
t_impact = 5.0
d_impact = 0.1

# Fall detection
fall_counter = 0
fall_threshold = 10
robot_fallen = False
fall_time = None


# ==========================================================
# 2. MPC INITIALIZATION
# ==========================================================

mpc = ZMP.LinearInvertedPendulumMPC(N_horizon, T_step, h_com, g)


# ==========================================================
# 3. FOOTSTEP GENERATION
# ==========================================================

footstep_generator = CandidateFootstepGenerator(l)
footsteps = footstep_generator.generate_full_sequence(
    start_pose,
    v_command,
    total_duration + T_step * N_horizon,
)

steps = [footsteps[k]["pos"][0:2] for k in range(len(footsteps))]


# ==========================================================
# 4. EXTERNAL FORCE DISCRETIZATION
# ==========================================================

k_start = int(t_impact / T_step)
k_end = k_start + int(d_impact / T_step)

Bd = np.array([
    [0.5 * T_step**2],
    [T_step],
    [0.0],
])


# ==========================================================
# 5. ZMP CONSTRAINTS FROM ROTATED FOOTRECTANGLES
# ==========================================================

def create_rotated_zmp_constraints(x_ref, y_ref, theta, foot_size_fwd, foot_size_lat):
    """
    Compute ZMP min and max bounds using a rotated rectangular foot model.
    The rotated rectangle is projected onto the global X and Y axes.
    """

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    corners_local = np.array([
        [-foot_size_fwd, -foot_size_lat],
        [foot_size_fwd, -foot_size_lat],
        [foot_size_fwd, foot_size_lat],
        [-foot_size_fwd, foot_size_lat],
    ])

    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    corners_global = corners_local @ R.T + np.array([x_ref, y_ref])

    x_min = corners_global[:, 0].min()
    x_max = corners_global[:, 0].max()
    y_min = corners_global[:, 1].min()
    y_max = corners_global[:, 1].max()

    return x_min, x_max, y_min, y_max


full_refs_lat = []
full_z_min_lat = []
full_z_max_lat = []

full_refs_fwd = []
full_z_min_fwd = []
full_z_max_fwd = []

full_teta = []
indicator = []

for step in footsteps:

    x_ref = step["pos"][0]
    y_ref = step["pos"][1]
    theta = step["pos"][2]

    x_min, x_max, y_min, y_max = create_rotated_zmp_constraints(
        x_ref, y_ref, theta, foot_size_fwd, foot_size_lat
    )

    for _ in range(int(step["duration"] / T_step)):

        full_refs_lat.append(y_ref)
        full_z_min_lat.append(y_min)
        full_z_max_lat.append(y_max)

        full_refs_fwd.append(x_ref)
        full_z_min_fwd.append(x_min)
        full_z_max_fwd.append(x_max)

        indicator.append(step["side"])
        full_teta.append(theta)

steps_lat = full_refs_lat
steps_fwd = full_refs_fwd


# ==========================================================
# 6. MPC SIMULATION LOOP
# ==========================================================

x_lat = np.array([steps_lat[0], 0.0, 0.0])
x_fwd = np.array([steps_fwd[0], 0.0, 0.0])

history_com = []
history_zmp = []

iterations_simu = int(total_duration / T_step)
print(f"Starting simulation ({iterations_simu} iterations)...")

for k in tqdm(range(len(steps_lat) - N_horizon)):

    a_ext_lat = 0.0
    a_ext_fwd = 0.0

    if k_start <= k < k_end:
        a_ext_lat = F_lat / m_robot
        a_ext_fwd = F_fwd / m_robot

    # --- LATERAL AXIS ---

    refs_lat = steps_lat[k : k + N_horizon]
    z_min_lat = full_z_min_lat[k : k + N_horizon]
    z_max_lat = full_z_max_lat[k : k + N_horizon]

    jerks_lat = mpc.solve(x_lat, refs_lat, z_min_lat, z_max_lat)
    u_lat = jerks_lat[0]

    x_lat = (
        mpc.A @ x_lat
        + (mpc.B.flatten() * u_lat)
        + (Bd.flatten() * a_ext_lat)
    )

    # --- FORWARD AXIS ---

    refs_fwd = steps_fwd[k : k + N_horizon]
    z_min_fwd = full_z_min_fwd[k : k + N_horizon]
    z_max_fwd = full_z_max_fwd[k : k + N_horizon]

    jerks_fwd = mpc.solve(x_fwd, refs_fwd, z_min_fwd, z_max_fwd)
    u_fwd = jerks_fwd[0]

    x_fwd = (
        mpc.A @ x_fwd
        + (mpc.B.flatten() * u_fwd)
        + (Bd.flatten() * a_ext_fwd)
    )

    # --- ZMP COMPUTATION ---

    zmp_x = (
        x_fwd[0]
        - (h_com / g) * x_fwd[2]
        + (h_com / g) * a_ext_fwd
    )

    zmp_y = (
        x_lat[0]
        - (h_com / g) * x_lat[2]
        + (h_com / g) * a_ext_lat
    )

    # --- SUPPORT POLYGON CHECK ---

    inside_support = (
        (z_min_lat[0] <= zmp_y <= z_max_lat[0])
        and (z_min_fwd[0] <= zmp_x <= z_max_fwd[0])
    )

    if not inside_support:
        fall_counter += 1
    else:
        fall_counter = 0

    if fall_counter >= fall_threshold and not robot_fallen:
        robot_fallen = True
        fall_time = k * T_step
        print(f"\nROBOT FALL DETECTED at t = {fall_time:.2f} s")

    history_com.append([x_fwd[0], x_lat[0], h_com])
    history_zmp.append([zmp_x.item(), zmp_y.item()])


# ==========================================================
# 7. VISUALIZATION
# ==========================================================

import vizualisation

time = np.arange(len(history_com)) * T_step
history_ref = [
    [full_refs_fwd[k], full_refs_lat[k]]
    for k in range(len(history_com))
]

vizualisation.plot_2d_simulation(
    time,
    history_com,
    history_zmp,
    history_ref,
    full_z_max_lat,
    full_z_min_lat,
)

vizualisation.run_robot_visualization(
    history_ref,
    indicator,
    history_com,
    history_zmp,
    foot_size_lat,
    foot_size_fwd,
    full_teta,
    T_step,
    v_max,
)
