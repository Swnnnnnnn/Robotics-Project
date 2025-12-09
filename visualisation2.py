import matplotlib.pyplot as plt
import numpy as np
import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf
from tqdm import tqdm
import time as time_module


# ==========================================================
# ===================== 2D PLOT ============================
# ==========================================================

def plot_2d_simulation(time, history_com, history_zmp, history_ref, full_z_max_lat, full_z_min_lat):
    """
    2D plot : projection sur l'axe LATÉRAL.
    Compatible avec :
    - ancien mode 1D
    - nouveau mode 2D (avant + latéral)
    """

    history_com = np.array(history_com)
    history_zmp = np.array(history_zmp)
    history_ref = np.array(history_ref)

    # Si on est en 2D, on prend la coordonnée latérale = y
    if history_com.shape[1] == 3:
        com_lat = history_com[:, 1]
        zmp_lat = history_zmp[:, 1]
        ref_lat = history_ref[:, 1]
    else:
        com_lat = history_com[:, 0]
        zmp_lat = history_zmp.flatten()
        ref_lat = history_ref

    plt.figure(figsize=(10, 5))
    plt.plot(time, ref_lat, 'k--', label='Centre Pied (Ref)')
    plt.plot(time, com_lat, 'b-', linewidth=2, label='CoM Position')
    plt.plot(time, zmp_lat, 'g-', alpha=0.7, label='ZMP')
    plt.fill_between(time,
                     full_z_min_lat[0:len(time)],
                     full_z_max_lat[0:len(time)],
                     color='gray', alpha=0.2, label='Contraintes Pieds')

    plt.title("MPC - Marche humanoïde (projection latérale)")
    plt.xlabel("Temps (s)")
    plt.ylabel("Position latérale (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig('test_plot.png')
    plt.show()


# ==========================================================
# ===================== 3D ANIMATION =======================
# ==========================================================

def plot_3d_simulation(history_com, history_zmp, history_ref, foot_size_fwd, foot_size_lat, full_teta, T_step,indicator = None):
    """
    Animation 3D complète :
    - déplacement AVANT + LATÉRAL
    - CoM, ZMP, pied mobile
    """

    viewer = meshcat.Visualizer()
    viewer.open()

    # ---------- OBJETS ----------
    viewer["com"].set_object(
        geom.Sphere(0.03),
        geom.MeshLambertMaterial(color=0xff0000)
    )

    viewer["zmp"].set_object(
        geom.Sphere(0.02),
        geom.MeshLambertMaterial(color=0x0000ff)
    )

    viewer["ground"].set_object(
        geom.Box([10.0, 10.0, 0.0001]),
        geom.MeshLambertMaterial(color=0xaaaaaa, opacity=0.5)
    )
    viewer["ground"].set_transform(
        tf.translation_matrix([0, 0, -0.0005])
    )

    print("Meshcat viewer initialized. Navigate to the provided URL.")

    # ---------- DONNÉES ----------
    history_com_arr = np.array(history_com)
    history_zmp_arr = np.array(history_zmp)
    history_ref_arr = np.array(history_ref)

    print(f"Animating {len(history_com_arr)} frames...")

    # ---------- ANIMATION ----------
    flag = 0
    for i in tqdm(range(len(history_com_arr) - 1)):

        com_x = float(history_com_arr[i, 0])
        com_y = float(history_com_arr[i, 1])
        com_z = float(history_com_arr[i, 2])

        zmp_x = float(history_zmp_arr[i, 0])
        zmp_y = float(history_zmp_arr[i, 1])

        ref_x = float(history_ref_arr[i, 0])
        ref_y = float(history_ref_arr[i, 1])

        # -------- CoM --------
        viewer["com"].set_transform(
            tf.translation_matrix([com_x, com_y, com_z])
        )

        # -------- ZMP --------
        viewer["zmp"].set_transform(
            tf.translation_matrix([zmp_x, zmp_y, 0.0])
        )

        # -------- PIED --------
        # Dimensions du pied
        foot_width_x = float(2 * foot_size_fwd)
        foot_width_y = float(2 * foot_size_lat)
        
        # Position de référence
        foot_center_x = ref_x 
        foot_center_y = ref_y

        # Orientation
        theta = float(full_teta[i])
        
        # Créer une transformation avec rotation autour de Z
        rotation_matrix = tf.rotation_matrix(theta, [0, 0, 1])
        translation_matrix = tf.translation_matrix([foot_center_x, foot_center_y, 0.005])
        transform = np.dot(translation_matrix, rotation_matrix)

        if indicator:
            if indicator[i] == 1:
                foot_color = 0x00ff00  # Green for right foot
            else:
                foot_color = 0xff0000  # Red for left foot
        else:
            foot_color = 0x00ff00  # Default color
        viewer["foot"].set_object(
            geom.Box([foot_width_x, foot_width_y, 0.01]),
            geom.MeshLambertMaterial(color=foot_color, opacity=0.35)
        )

        viewer["foot"].set_transform(transform)

        time_module.sleep(T_step)

    print("Animation finished.")
