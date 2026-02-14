class KalmanFilter:
    """1D Kalman filter for tracking accuracy convergence in K-FL.

    Based on Algorithm 1 in:
    H. Kim et al., "K-FL: Kalman Filter-Based Clustering Federated
    Learning Method", IEEE Access, 2023.

    System model:
        x_t = A * x_{t-1} + w_{t-1},  w_t ~ N(0, Q)
        z_t = H * x_t + v_t,           v_t ~ N(0, R)
    """

    def __init__(self, A=1.0, H=1.0, Q=0.0179, R=0.0003, x_hat=0.0, P=0.001):
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R
        self.x_hat = x_hat
        self.P = P
        self.x_hat_minus = None
        self.P_minus = None

    def predict(self):
        """Step 3: Predict estimation value and error covariance."""
        self.x_hat_minus = self.A * self.x_hat
        self.P_minus = self.A * self.P * self.A + self.Q
        return self.x_hat_minus

    def update(self, z):
        """Steps 4-6: Kalman gain, estimation value, error covariance."""
        K = (self.P_minus * self.H) / (self.H * self.P_minus * self.H + self.R)
        self.x_hat = self.x_hat_minus + K * (z - self.H * self.x_hat_minus)
        self.P = self.P_minus - K * self.H * self.P_minus
        return self.x_hat
