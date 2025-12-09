import numpy as np


class CandidateFootstepGenerator:
    """
    Candidate footstep generator based on a simple template-based walking model
    with kinematic and geometric constraints.
    """

    def __init__(self, lateral_distance):
        """
        Initialize the footstep generator.

        Parameters
        ----------
        lateral_distance : float
            Nominal lateral distance between left and right feet.
        """

        # Template model parameters (Paper Section III-A)
        self.v_bar = 0.2
        self.T_bar = 0.8
        self.alpha = 0.1
        self.l = lateral_distance

        # Kinematic constraints (Paper Section III-B)
        self.d_ax = 0.3
        self.d_ay = 0.15

    def compute_step_duration(self, v_norm):
        """
        Compute adaptive step duration based on velocity norm.

        Parameters
        ----------
        v_norm : float
            Norm of the commanded planar velocity.

        Returns
        -------
        Ts : float
            Step duration.
        """

        denom = self.alpha + v_norm
        if denom < 1e-4:
            denom = 1e-4

        return self.T_bar * (self.alpha + self.v_bar) / denom

    def get_rotation_matrix(self, theta):
        """
        Return the 2D rotation matrix for a given angle.

        Parameters
        ----------
        theta : float
            Rotation angle.

        Returns
        -------
        R : ndarray, shape (2, 2)
            Rotation matrix.
        """

        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def generate_full_sequence(self, start_pose, v_cmd, duration):
        """
        Generate a full sequence of candidate footsteps.

        Parameters
        ----------
        start_pose : array_like, shape (3,)
            Initial foot pose [x, y, theta] assumed to be the right foot.
        v_cmd : array_like, shape (3,)
            Commanded velocity [vx, vy, omega].
        duration : float
            Total walking duration.

        Returns
        -------
        footsteps : list of dict
            Each element contains:
                - 'pos' : ndarray (x, y, theta)
                - 'time' : float
                - 'duration' : float
                - 'side' : int (+1 left, -1 right)
        """

        footsteps = []
        current_time = 0.0

        # Template robot body state (initialized at mid-feet approximation)
        current_pose = np.array(start_pose, dtype=float)

        # The first foot is assumed to be the right one
        previous_foot_pose = np.array(start_pose, dtype=float)

        # Store initial footstep
        footsteps.append({
            "pos": previous_foot_pose,
            "time": 0.0,
            "duration": 1.0,
            "side": -1,
        })

        # Next foot will be the left one
        next_foot_side = 1

        while current_time < duration:
            vx, vy, w = v_cmd
            v_norm = np.sqrt(vx**2 + vy**2)

            # 1. Step timing
            Ts = self.compute_step_duration(v_norm)

            # 2. Integration of the template model (robot body motion)
            dtheta = w * Ts
            R_robot = self.get_rotation_matrix(current_pose[2])
            delta_world = R_robot @ np.array([vx * Ts, vy * Ts])

            current_pose[0] += delta_world[0]
            current_pose[1] += delta_world[1]
            current_pose[2] += dtheta

            # 3. Ideal foot placement in the robot frame
            R_next = self.get_rotation_matrix(current_pose[2])
            offset_lateral = np.array([0.0, next_foot_side * self.l / 2.0])

            ideal_foot_pos_world = current_pose[:2] + (R_next @ offset_lateral)
            ideal_foot_theta = current_pose[2]

            # 4. Constraint enforcement (clipping in the previous foot frame)
            R_prev = self.get_rotation_matrix(previous_foot_pose[2])
            R_prev_T = R_prev.T

            diff_world = ideal_foot_pos_world - previous_foot_pose[:2]
            diff_local = R_prev_T @ diff_world

            # Sagittal constraint (step length bound)
            diff_local[0] = np.clip(
                diff_local[0], -self.d_ax / 2.0, self.d_ax / 2.0
            )

            # Coronal constraint (lateral placement bound)
            nominal_y = next_foot_side * self.l
            min_y = nominal_y - self.d_ay / 2.0
            max_y = nominal_y + self.d_ay / 2.0
            if min_y > max_y:
                min_y, max_y = max_y, min_y

            diff_local[1] = np.clip(diff_local[1], min_y, max_y)

            # 5. Final validated foot pose
            final_pos_world = previous_foot_pose[:2] + (R_prev @ diff_local)

            theta_max_step = np.pi / 8.0
            delta_theta = ideal_foot_theta - previous_foot_pose[2]

            delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
            delta_theta = np.clip(delta_theta, -theta_max_step, theta_max_step)

            final_theta = previous_foot_pose[2] + delta_theta

            new_step_pose = np.array([
                final_pos_world[0],
                final_pos_world[1],
                final_theta,
            ])

            # Store footstep
            current_time += Ts
            footsteps.append({
                "pos": new_step_pose,
                "time": current_time,
                "duration": Ts,
                "side": next_foot_side,
            })

            # Prepare next iteration
            previous_foot_pose = new_step_pose
            next_foot_side *= -1

        return footsteps
