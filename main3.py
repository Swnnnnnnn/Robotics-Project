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
h_com = 0.84
l = 0.35
foot_size_lat = 0.05 - 0.02
foot_size_fwd = 0.07 - 0.02
m_robot = 95.0
v_max = 0.5

# Commanded motion
v_command = [0.5, 0.0, 0.0]    # [vx, vy, omega]
total_duration = 6.5
start_pose = [0.0, 0, 0.0]     # Initial right foot pose [x, y, theta]

# External disturbance parameters
F_lat = 150.0
F_fwd = 0.0
t_impact = 7
d_impact = 0.1

# Centrifugal force parameters
enable_centrifugal_force = False  # Enable/disable centrifugal force calculation

# Fall detection
fall_counter = 0
fall_threshold = 10
robot_fallen = False
fall_time = None

# ==========================================================
# 2. MPC INITIALIZATION
# ==========================================================

# Utiliser le solveur 2D avec rectangles tournés
mpc = ZMP.LinearInvertedPendulumMPC2D(N_horizon, T_step, h_com, g)



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
        [-foot_size_fwd, foot_size_lat]
    ])

    # Rotation matrix
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # Transform corners to global frame
    corners_global = corners_local @ R.T + np.array([x_ref, y_ref])

    # Get min and max in global frame
    x_min = corners_global[:, 0].min()
    x_max = corners_global[:, 0].max()
    y_min = corners_global[:, 1].min()
    y_max = corners_global[:, 1].max()

    return x_min, x_max, y_min, y_max


steps_lat = []
full_z_min_lat = []
full_z_max_lat = []

steps_fwd = []
full_z_min_fwd = []
full_z_max_fwd = []

full_teta = []
indicator = []

flat=2 #first step special case
for pas in footsteps:

    x_ref = pas['pos'][0]
    y_ref = pas['pos'][1]
    teta = pas['pos'][2]
    x_min_lat, x_max_lat, y_min_lat, y_max_lat = create_rotated_zmp_constraints(
            x_ref, y_ref, teta, flat*foot_size_fwd, flat*foot_size_lat)
    flat=1
    for _ in range(int(pas['duration'] / T_step)):

    # Lateral Axis (Y)
        steps_lat.append(y_ref)
        full_z_min_lat.append(y_min_lat)
        full_z_max_lat.append(y_max_lat)
    
    # Forward Axis (X)
        steps_fwd.append(x_ref)
        full_z_min_fwd.append(x_min_lat)
        full_z_max_fwd.append(x_max_lat)

        indicator.append(pas['side'])
        full_teta.append(teta)


# ==========================================================
# 6. MPC SIMULATION LOOP
# ==========================================================

x_lat = np.array([start_pose[1], 0.0, 0.0])
x_fwd = np.array([start_pose[0], 0.0, 0.0])

history_com = []
history_zmp = []

iterations_simu = int(total_duration / T_step)

print(f"Lancement de la simulation ({iterations_simu} itérations)...")

for k in tqdm(range(len(steps_lat) - N_horizon)):

    # acceleration due to external force
    a_ext_lat = 0.0
    a_ext_fwd = 0.0

    if k_start <= k < k_end:
        a_ext_lat = F_lat / m_robot
        a_ext_fwd = F_fwd / m_robot

    # --- CENTRIFUGAL FORCE ---
    # F_centrifugal = m * omega^2 * r
    # In the global frame, this appears as a lateral acceleration
    # pointing outward from the center of rotation
    if enable_centrifugal_force:
        # Commanded angular velocity
        omega = v_command[2]
        
        # Current velocity in the global frame
        v_fwd = x_fwd[1]
        v_lat = x_lat[1]
        
        # Velocity magnitude (distance from center of rotation)
        v_norm = np.sqrt(v_fwd**2 + v_lat**2)
        
        # Centrifugal acceleration magnitude: a_c = omega^2 * r = omega^2 * (v / omega) = omega * v
        if abs(omega) > 1e-6 and v_norm > 1e-6:

            # Centrifugal acceleration in global frame
            a_centrifugal_fwd = v_lat * omega
            a_centrifugal_lat = -v_fwd * omega
                
            a_ext_fwd += a_centrifugal_fwd
            a_ext_lat += a_centrifugal_lat
    # --- 2D MPC RESOLUTION ---
    # Préparer les données pour le solveur 2D
    
    # current state [x_fwd_pos, x_fwd_vel, x_fwd_acc, x_lat_pos, x_lat_vel, x_lat_acc]
    x_current_2d = np.hstack([x_fwd, x_lat])
    

    refs_2d = np.array([[steps_fwd[k + i], steps_lat[k + i]] for i in range(N_horizon)])
    
    foot_positions = np.array([[steps_fwd[k + i], steps_lat[k + i]] for i in range(N_horizon)])
    foot_theta = np.array([full_teta[k + i] for i in range(N_horizon)])
    
    if (k+N_horizon)//T_step < 2:
        first_call = True
    else:
        first_call = False
    # Solve the 2D MPC problem
    U_optimal_2d = mpc.solve(
        x_current_2d,
        refs_2d,
        foot_positions,
        foot_theta,
        foot_size_fwd,
        foot_size_lat,
        first_call=first_call
    )
    
    u_fwd = U_optimal_2d[0]
    u_lat = U_optimal_2d[N_horizon]
    
    # --- STATE UPDATE ---
    x_fwd = (
        mpc.A @ x_fwd
        + (mpc.B.flatten() * u_fwd)
        + (Bd.flatten() * a_ext_fwd)
    )
    
    x_lat = (
        mpc.A @ x_lat
        + (mpc.B.flatten() * u_lat)
        + (Bd.flatten() * a_ext_lat)
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

    # --- SUPPORT POLYGON CHECK  ---
    theta_k = float(full_teta[k])
    x_ref_k = float(steps_fwd[k])
    y_ref_k = float(steps_lat[k])
    
    # ZMP in the foot frame (centered at [x_ref_k, y_ref_k])
    zmp_rel_x = zmp_x - x_ref_k
    zmp_rel_y = zmp_y - y_ref_k
    
    # ZMP in the local foot frame (inverse rotations)
    zmp_local_x = np.cos(theta_k) * zmp_rel_x + np.sin(theta_k) * zmp_rel_y
    zmp_local_y = -np.sin(theta_k) * zmp_rel_x + np.cos(theta_k) * zmp_rel_y
    
    inside_support = (
        abs(zmp_local_x) <= foot_size_fwd and
        abs(zmp_local_y) <= foot_size_lat
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
history_ref = [[steps_fwd[k], steps_lat[k]] for k in range(len(history_com))]
vizualisation.plot_2d_simulation(time, history_com, history_zmp, history_ref, full_z_max_lat, full_z_min_lat, full_teta, foot_size_fwd, foot_size_lat)
vizualisation.plot_top_view_trajectory(history_com, history_zmp, history_ref, full_teta, foot_size_fwd, foot_size_lat, indicator)
# vizualisation.run_robot_visualization(history_ref, indicator, history_com, history_zmp, foot_size_lat, foot_size_fwd, full_teta, T_step,v_max)

# vizualisation2.plot_3d_simulation(history_com, history_zmp, history_ref,
#                                   foot_size_fwd, foot_size_lat, full_teta,
#                                    T_step,indicator)
