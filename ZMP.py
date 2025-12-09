import numpy as np
from qpsolvers import solve_qp


class LinearInvertedPendulumMPC:
    """
    Model Predictive Control (MPC) for the Linear Inverted Pendulum Model (LIPM)
    based on ZMP regulation with jerk as control input.

    State vector:
        x = [position, velocity, acceleration]

    Control input:
        u = jerk
    """

    def __init__(self, horizon_length, T, com_height, gravity):
        """
        Initialize the LIPM-based MPC controller.

        Parameters
        ----------
        horizon_length : int
            Prediction horizon length.
        T : float
            Sampling period.
        com_height : float
            Constant height of the center of mass.
        gravity : float
            Gravitational acceleration.
        """

        self.T = T
        self.N = horizon_length
        self.h = com_height
        self.g = gravity

        # Cost function weights
        self.Q = 1.0     # ZMP tracking weight
        self.R = 1e-4    # Jerk regularization weight

        # Discrete-time state-space matrices (exact discretization)
        # State = [position, velocity, acceleration]
        self.A = np.array([
            [1.0, T, T**2 / 2.0],
            [0.0, 1.0, T],
            [0.0, 0.0, 1.0],
        ])

        self.B = np.array([
            [T**3 / 6.0],
            [T**2 / 2.0],
            [T],
        ])

        # ZMP output equation: z = x - (h / g) * ddx
        self.C = np.array([[1.0, 0.0, -com_height / gravity]])

        # Precompute prediction matrices
        self._build_prediction_matrices()
        self._build_final_state_matrices()

    def _build_prediction_matrices(self):
        """
        Construct the prediction matrices Px and Pu such that:

            Z_future = Px @ x_current + Pu @ U
        """

        self.Px = np.zeros((self.N, 3))
        self.Pu = np.zeros((self.N, self.N))

        for i in range(self.N):
            self.Px[i, :] = self.C @ np.linalg.matrix_power(self.A, i + 1)

            for j in range(i + 1):
                self.Pu[i, j] = (
                    self.C
                    @ np.linalg.matrix_power(self.A, i - j)
                    @ self.B
                ).item()

    def _build_final_state_matrices(self):
        """
        Construct matrices for predicting the final state:

            X_N = A^N x_0 + Su_final U
        """

        self.A_power_N = np.linalg.matrix_power(self.A, self.N)

        self.Su_final = np.zeros((3, self.N))

        for j in range(self.N):
            power = self.N - 1 - j
            self.Su_final[:, j] = (
                np.linalg.matrix_power(self.A, power) @ self.B
            ).flatten()

    def solve(self, x_current, zmp_ref_window, zmp_min_window, zmp_max_window):
        """
        Solve the quadratic program to compute the optimal jerk sequence.

        Parameters
        ----------
        x_current : ndarray, shape (3,)
            Current state [position, velocity, acceleration].
        zmp_ref_window : array_like, shape (N,)
            Reference ZMP trajectory.
        zmp_min_window : array_like, shape (N,)
            Minimum ZMP constraint.
        zmp_max_window : array_like, shape (N,)
            Maximum ZMP constraint.

        Returns
        -------
        U_optimal : ndarray, shape (N,)
            Optimal jerk sequence over the horizon.
        """

        x_current = x_current.reshape(3, 1)
        z_ref = np.asarray(zmp_ref_window).reshape(self.N, 1)
        z_min = np.asarray(zmp_min_window).reshape(self.N, 1)
        z_max = np.asarray(zmp_max_window).reshape(self.N, 1)

        # Quadratic cost
        H = self.Q * (self.Pu.T @ self.Pu) + self.R * np.eye(self.N)

        tracking_error = self.Px @ x_current - z_ref
        g = self.Q * self.Pu.T @ tracking_error

        # Inequality constraints: ZMP bounds
        G = np.vstack([self.Pu, -self.Pu])

        h_upper = z_max - self.Px @ x_current
        h_lower = -z_min + self.Px @ x_current
        h = np.vstack([h_upper, h_lower])

        # Stability constraint using DCM at the end of the horizon
        omega = np.sqrt(self.g / self.h)

        C_dcm = np.array([1.0, 1.0 / omega, 0.0])

        A_eq = (C_dcm @ self.Su_final).reshape(1, self.N)

        ref_final = z_ref[-1, 0]

        x_final_free = self.A_power_N @ x_current
        dcm_final_free = C_dcm @ x_final_free

        b_eq = np.array([ref_final - dcm_final_free])

        # QP resolution
        U_optimal = solve_qp(
            H,
            g.flatten(),
            G,
            h.flatten(),
            A=A_eq,
            b=b_eq,
            solver="osqp",
        )

        if U_optimal is None:
            return np.zeros(self.N)

        return U_optimal
