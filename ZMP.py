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
        self.Q = 100.0     # ZMP tracking weight
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

from numpy.linalg import matrix_power


class LinearInvertedPendulumMPC2D:
    """
    2D MPC for Linear Inverted Pendulum Model with coupled X-Y control.
    
    This version implements the Intrinsically Stable MPC (IS-MPC) framework presented in 
    "MPC for Humanoid Gait Generation: Stability and Feasibility" .
    
    It enforces the stability constraint to guarantee that the Center of Mass (CoM) 
    trajectory remains bounded with respect to the ZMP trajectory .
    """
    
    def __init__(self, horizon_length, T, com_height, gravity):
        """
        Initialize the 2D LIPM-based MPC controller.
        
        Parameters
        ----------
        horizon_length : int
           Prediction horizon length 
        T : float
            Sampling period (delta in the paper )
        com_height : float
            Constant height of the center of mass 
        gravity : float
            Gravitational acceleration 
        """
        
        self.T = T
        self.N = horizon_length
        self.h = com_height
        self.g = gravity
        
        # Cost function weights (implied by standard MPC formulation)
        self.Q = 1     # Weight for ZMP tracking error
        self.R = 1e-4  # Weight for Jerk minimization (Control input)
        
        # --- Prediction Model: Dynamically Extended LIP ---
        # The paper uses a LIP model with ZMP velocity as input, resulting in a 3rd order system 
        # (position, velocity, acceleration/ZMP) .
        # State vector x = [c, c_dot, z_mp] (mapped to pos, vel, acc here).
        # See Equation (4)
        
        # Discrete-time state matrix A (Exact discretization of Eq 4)
        self.A = np.array([
            [1.0, T, T**2 / 2.0],
            [0.0, 1.0, T],
            [0.0, 0.0, 1.0],
        ])
        
        # Discrete-time control matrix B (Input is Jerk/ZMP velocity)
        self.B = np.array([
            [T**3 / 6.0],
            [T**2 / 2.0],
            [T],
        ])
        
        # Output matrix C to extract ZMP.
        # In the LIP model, x_z = x_c - (h/g) * x_c_ddot

        self.C = np.array([[1.0, 0.0, -com_height / gravity]])
        
        #Construct block-diagonal prediction matrices for the 2D coupled system.
        #Since LIP dynamics are decoupled in X and Y], we can block-diagonalize.
        
        # 1D prediction matrices
        Px_1d = np.zeros((self.N, 3))
        Pu_1d = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            Px_1d[i, :] = self.C @ matrix_power(self.A, i + 1)
            
            for j in range(i + 1):
                Pu_1d[i, j] = (
                    self.C
                    @ matrix_power(self.A, i - j)
                    @ self.B
                ).item()
        
        # 2D block-diagonal versions (X states then Y states)
        self.Px_2d = np.block([
            [Px_1d, np.zeros((self.N, 3))],
            [np.zeros((self.N, 3)), Px_1d]
        ])  # shape (2N, 6)
        
        self.Pu_2d = np.block([
            [Pu_1d, np.zeros((self.N, self.N))],
            [np.zeros((self.N, self.N)), Pu_1d]
        ])  # shape (2N, 2N)
    
        #Construct matrices to predict the final state at step k+C (end of horizon).
        #Required for the Stability Terminal Constraint.
        
        A_power_N = matrix_power(self.A, self.N)
        
        Su_final_1d = np.zeros((3, self.N))
        for j in range(self.N):
            power = self.N - 1 - j
            Su_final_1d[:, j] = (
                matrix_power(self.A, power) @ self.B
            ).flatten()
        
        self.A_power_N_2d = np.block([
            [A_power_N, np.zeros((3, 3))],
            [np.zeros((3, 3)), A_power_N]
        ])  # shape (6, 6)

        self.Su_final_2d = np.block([
            [Su_final_1d, np.zeros((3, self.N))],
            [np.zeros((3, self.N)), Su_final_1d]
        ])  # shape (6, 2N)
    
    def solve(self, x_current, zmp_ref_window, foot_positions, foot_theta, 
              foot_size_fwd, foot_size_lat , first_call=False):
        """
        Solve the IS-MPC QP problem for 2D LIPM.
        
        Parameters match the QP formulation in Section VI-A
        """
        if first_call:
            # On the first call, we may want to initialize differently
            foot_size_fwd *= 2.0
        x_current = x_current.reshape(6, 1)
        refs_arr = np.asarray(zmp_ref_window)
        # On empile la colonne 0 (X) puis la colonne 1 (Y)
        zmp_ref = np.concatenate([refs_arr[:, 0], refs_arr[:, 1]]).reshape(2 * self.N, 1)
        
        # --- QUADRATIC COST ---
        # Minimizes ZMP error and Control effort (Jerk).
        H = self.Q * (self.Pu_2d.T @ self.Pu_2d) + self.R * np.eye(2 * self.N)
        
        tracking_error = self.Px_2d @ x_current - zmp_ref
        g = self.Q * self.Pu_2d.T @ tracking_error
        
        # --- INEQUALITY CONSTRAINTS: ZMP bounds ---
        # Enforces that ZMP lies within the support polygon
        # Use Equation (6) 
        
        G_list = []
        h_list = []
        
        for k in range(self.N):
            # The support region is a rectangle centered at the footstep.
            x_center = float(foot_positions[k, 0])
            y_center = float(foot_positions[k, 1])
            theta = float(foot_theta[k])
            
            # Compute normals for the rotated footstep rectangle.
            # This implements the logic described for "admissible region".
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            normal_fwd = np.array([cos_t, sin_t])
            normal_lat = np.array([-sin_t, cos_t]) 
            
            # Constraints are linear inequalities derived from geometry
            constraints_k = [
                (normal_fwd, foot_size_fwd),
                (-normal_fwd, foot_size_fwd),
                (normal_lat, foot_size_lat),
                (-normal_lat, foot_size_lat),
            ]
            
            for (nx, ny), d in constraints_k:
                px_k_fwd = self.Px_2d[k, :]      
                px_k_lat = self.Px_2d[self.N + k, :]  
                
                pu_k_fwd = self.Pu_2d[k, :]      
                pu_k_lat = self.Pu_2d[self.N + k, :]  
                
                # Formulate G*U <= h for this edge
                g_ineq = nx * pu_k_fwd + ny * pu_k_lat  
                
                z_free_k = nx * (px_k_fwd @ x_current) + ny * (px_k_lat @ x_current)
                h_ineq = d + (nx * x_center + ny * y_center) - z_free_k.item()
                
                G_list.append(g_ineq)
                h_list.append(h_ineq)
        
        G = np.vstack(G_list)
        h = np.array(h_list).reshape(-1, 1)

        # --- IS-MPC STABILITY CONSTRAINT (Equality) ---
        # This is the core contribution of the paper.
        # It links the state at the end of the horizon to a stable future.
        
        # 1. Define DCM (Divergent Component of Motion)
        
        # Projection matrix from State to DCM: x_u = x_c + x_dot_c / eta
        # See Equation (9)
        C_dcm = np.array([1.0, 1.0 / np.sqrt(self.g / self.h), 0.0])
        C_dcm_2d = np.block([[C_dcm, np.zeros(3)],
                              [np.zeros(3), C_dcm]]) 
        
        # 2. Formulate the Terminal Constraint
        # We enforce DCM_final = CoM_final (instead of ZMP_ref_final)
        # This ensures the robot is in a stable state at the end of the horizon
        
        # Left Hand Side: DCM at k+C predicted by the MPC control variables U
        A_eq = C_dcm_2d @ self.Su_final_2d  
        
        
        # Target: DCM_final should equal the reference footstep position (not ZMP)
        # This creates a more centered trajectory
        foot_ref_final = foot_positions[-1, :]  # Last footstep position
        
        # Free response of DCM (due to current state x_k)
        x_final_free = self.A_power_N_2d @ x_current 
        dcm_final_free = C_dcm_2d @ x_final_free 
        
        # We want: DCM_final = foot_ref_final
        # DCM_final = dcm_final_free + C_dcm_2d @ Su_final_2d @ U
        # So: C_dcm_2d @ Su_final_2d @ U = foot_ref_final - dcm_final_free
        b_eq = (foot_ref_final.reshape(2, 1) - dcm_final_free).flatten() 
        
        # --- SOLVE QP ---
        # Solve the Quadratic Program
        U_optimal = solve_qp(
            H.astype(np.double),
            g.flatten().astype(np.double),
            G.astype(np.double),
            h.flatten().astype(np.double),
            A=A_eq.astype(np.double),
            b=b_eq.astype(np.double),
            solver="osqp",
        )

        if U_optimal is None:
            #print("Warning: QP with stability constraint failed.")
            # Fallback (may lose stability, but attempts to keep running)
            U_optimal = solve_qp(
                H.astype(np.double),
                g.flatten().astype(np.double),
                G.astype(np.double),
                h.flatten().astype(np.double),
                solver="osqp",
            )
            
            if U_optimal is None:
                return np.zeros(2 * self.N)
        
        return U_optimal