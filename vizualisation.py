import time
import numpy as np
import matplotlib.pyplot as plt

import example_robot_data as erd
from tqdm import tqdm

import meshcat.geometry as geom
import meshcat.transformations as tf

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer


def plot_2d_simulation(
    time,
    history_com,
    history_zmp,
    history_ref,
    full_z_max_lat,
    full_z_min_lat,
):
    """
    2D plot of the lateral motion (Y-axis projection).
    Compatible with both 1D and 2D MPC outputs.
    """

    history_com = np.array(history_com)
    history_zmp = np.array(history_zmp)
    history_ref = np.array(history_ref)

    if history_com.shape[1] == 3:
        com_lat = history_com[:, 1]
        zmp_lat = history_zmp[:, 1]
        ref_lat = history_ref[:, 1]
    else:
        com_lat = history_com[:, 0]
        zmp_lat = history_zmp.flatten()
        ref_lat = history_ref

    plt.figure(figsize=(10, 5))
    plt.plot(time, ref_lat, "k--", label="Foot Center (Reference)")
    plt.plot(time, com_lat, "b-", linewidth=2, label="CoM Position")
    plt.plot(time, zmp_lat, "g-", alpha=0.7, label="ZMP")

    plt.fill_between(
        time,
        full_z_min_lat[0:len(time)],
        full_z_max_lat[0:len(time)],
        color="gray",
        alpha=0.2,
        label="Foot Constraints",
    )

    plt.title("Humanoid Walking MPC (Lateral Projection)")
    plt.xlabel("Time (s)")
    plt.ylabel("Lateral Position (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_plot.png")
    plt.show()


def run_robot_visualization(
    history_ref,
    indicator,
    history_com,
    history_zmp,
    foot_size_lat,
    foot_size_fwd,
    full_teta,
    T_step,
    v_max,
):
    """
    Visualize humanoid walking using Pinocchio and Meshcat with inverse kinematics.
    """

    robot = erd.load("talos")
    model = robot.model
    collision_model = robot.collision_model
    visual_model = robot.visual_model

    viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    viz.display(robot.q0)

    viewer = viz.viewer

    viewer["com_marker"].set_object(
        geom.Sphere(0.03),
        geom.MeshLambertMaterial(color=0xFF0000),
    )

    viewer["zmp_marker"].set_object(
        geom.Sphere(0.025),
        geom.MeshLambertMaterial(color=0x0000FF),
    )

    viewer["ground"].set_object(
        geom.Box([10.0, 10.0, 0.0001]),
        geom.MeshLambertMaterial(color=0xAAAAAA, opacity=0.5),
    )

    viewer["ground"].set_transform(tf.translation_matrix([0, 0, -0.0005]))

    data = model.createData()
    q = robot.q0.copy()

    # IK parameters
    max_iters_ik = 20
    damping = 1e-4
    gain_com = 2.0
    gain_foot = 5.0
    dt = T_step

    left_foot_frame = "left_sole_link"
    right_foot_frame = "right_sole_link"

    left_frame_id = model.getFrameId(left_foot_frame)
    right_frame_id = model.getFrameId(right_foot_frame)

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    def limit_foot_speed(current_pos, target_pos, max_speed):
        diff = target_pos - current_pos
        dist = np.linalg.norm(diff)

        if dist > max_speed:
            return current_pos + (diff / dist) * max_speed

        return target_pos

    def rotation_error(R_target, R_current):
        return pin.log3(R_target @ R_current.T)

    def find_next_different_step(steps, current_index):
        current_step = steps[current_index]

        for idx in range(current_index + 1, len(steps)):
            if steps[idx] != current_step:
                return idx

        return len(steps) - 1

    def ik_iteration(
        q,
        com_target,
        left_pos_target,
        right_pos_target,
        which_foot,
        flag,
        yaw_target,
    ):

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        com = data.com[0]
        e_com = com_target - com

        Jcom = pin.jacobianCenterOfMass(model, data, q)

        J_list = [Jcom]
        err_list = [e_com]
        weights = [gain_com]

        T_left = data.oMf[left_frame_id]
        p_left = T_left.translation
        R_left = T_left.rotation
        J_left6 = pin.computeFrameJacobian(
            model, data, q, left_frame_id, pin.LOCAL_WORLD_ALIGNED
        )

        if which_foot == 0:
            pos_err = left_pos_target - p_left
            R_target = pin.utils.rotate("z", yaw_target)
            ori_err = rotation_error(R_target, R_left)
            e_left = np.hstack((pos_err, ori_err))
            J_left = J_left6[:6, :]

        else:
            if flag:
                left_pos_target[2] = 0.05
                pos_err = left_pos_target - p_left
                R_target = pin.utils.rotate("z", yaw_target)
                ori_err = rotation_error(R_target, R_left)
                e_left = np.hstack((pos_err, ori_err))
                J_left = J_left6[:6, :]
            else:
                pos_err = np.array([left_pos_target[0] - p_left[0]])
                e_left = pos_err
                J_left = J_left6[0:1, :]

        J_list.append(J_left)
        err_list.append(e_left)
        weights.append(gain_foot)

        T_right = data.oMf[right_frame_id]
        p_right = T_right.translation
        R_right = T_right.rotation
        J_right6 = pin.computeFrameJacobian(
            model, data, q, right_frame_id, pin.LOCAL_WORLD_ALIGNED
        )

        if which_foot == 1:
            pos_err = right_pos_target - p_right
            R_target = pin.utils.rotate("z", yaw_target)
            ori_err = rotation_error(R_target, R_right)
            e_right = np.hstack((pos_err, ori_err))
            J_right = J_right6[:6, :]

        else:
            if flag:
                right_pos_target[2] = 0.05
                pos_err = right_pos_target - p_right
                R_target = pin.utils.rotate("z", yaw_target)
                ori_err = rotation_error(R_target, R_right)
                e_right = np.hstack((pos_err, ori_err))
                J_right = J_right6[:6, :]
            else:
                pos_err = np.array([right_pos_target[0] - p_right[0]])
                e_right = pos_err
                J_right = J_right6[0:1, :]

        J_list.append(J_right)
        err_list.append(e_right)
        weights.append(gain_foot)

        weighted_J = [w * J for w, J in zip(weights, J_list)]
        weighted_err = [w * e for w, e in zip(weights, err_list)]

        J_stack = np.vstack(weighted_J)
        err_stack = np.hstack(weighted_err)

        JJ = J_stack.T @ J_stack
        rhs = J_stack.T @ err_stack

        lam = damping * np.eye(JJ.shape[0])
        dq = np.linalg.solve(JJ + lam, rhs)

        q_next = pin.integrate(model, q, dq * dt * 0.5)

        return q_next

    q_current = q.copy()
    which_foot = 0

    for k in tqdm(range(len(history_com))):

        com_target = history_com[k]

        current_step_lat = history_ref[k][1]
        current_step_fwd = history_ref[k][0]

        pin.forwardKinematics(model, data, q_current)
        pin.updateFramePlacements(model, data)

        current_left_pos = data.oMf[left_frame_id].translation.copy()
        current_right_pos = data.oMf[right_frame_id].translation.copy()

        left_pos_target = current_left_pos.copy()
        right_pos_target = current_right_pos.copy()

        flag = False

        if indicator[k] == 1:
            new_which_foot = 0
        else:
            new_which_foot = 1

        if new_which_foot != which_foot:
            which_foot = new_which_foot
            flag = True

        next_idx = find_next_different_step(history_ref, k)
        next_step_lat = history_ref[next_idx][1]
        next_step_fwd = history_ref[next_idx][0]

        if which_foot == 1:
            right_pos_target[:2] = [current_step_fwd, current_step_lat]
            right_pos_target[2] = 0.0

            swing_progress = 0.5
            left_target_x = current_left_pos[0] + (next_step_fwd - current_left_pos[0]) * swing_progress
            left_target_y = current_left_pos[1] + (next_step_lat - current_left_pos[1]) * swing_progress

            left_pos_target = limit_foot_speed(
                current_left_pos,
                np.array([left_target_x, left_target_y, 0.0]),
                v_max,
            )

        else:
            left_pos_target[:2] = [current_step_fwd, current_step_lat]
            left_pos_target[2] = 0.0

            swing_progress = 0.5
            right_target_x = current_right_pos[0] + (next_step_fwd - current_right_pos[0]) * swing_progress
            right_target_y = current_right_pos[1] + (next_step_lat - current_right_pos[1]) * swing_progress

            right_pos_target = limit_foot_speed(
                current_right_pos,
                np.array([right_target_x, right_target_y, 0.0]),
                v_max,
            )

        q_tmp = q_current.copy()

        for _ in range(max_iters_ik):
            q_new = ik_iteration(
                q_tmp,
                com_target,
                left_pos_target,
                right_pos_target,
                which_foot,
                flag,
                full_teta[k],
            )

            if np.linalg.norm(q_new - q_tmp) < 1e-6:
                q_tmp = q_new
                break

            q_tmp = q_new

        q_current = q_tmp

        viz.display(q_current)

        if k < len(history_zmp):
            zmp_x = float(history_zmp[k][0])
            zmp_y = float(history_zmp[k][1])
        else:
            zmp_x = 0.0
            zmp_y = 0.0

        viewer["com_marker"].set_transform(
            tf.translation_matrix([
                history_com[k][0],
                history_com[k][1],
                history_com[k][2],
            ])
        )

        viewer["zmp_marker"].set_transform(
            tf.translation_matrix([zmp_x, zmp_y, 0.01])
        )

        foot_width_x = float(2 * foot_size_fwd)
        foot_width_y = float(2 * foot_size_lat)

        foot_center_x = current_step_fwd
        foot_center_y = current_step_lat

        theta = float(full_teta[k])

        rotation_matrix = tf.rotation_matrix(theta, [0, 0, 1])
        translation_matrix = tf.translation_matrix(
            [foot_center_x, foot_center_y, 0.005]
        )

        transform = np.dot(translation_matrix, rotation_matrix)

        if indicator:
            if indicator[k] == 1:
                foot_color = 0x00FF00
            else:
                foot_color = 0xFF0000
        else:
            foot_color = 0x00FF00

        viewer["foot"].set_object(
            geom.Box([foot_width_x, foot_width_y, 0.01]),
            geom.MeshLambertMaterial(color=foot_color, opacity=0.35),
        )

        viewer["foot"].set_transform(transform)
