import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm

def wedge(phi):
    """
    R^3 vector to so(3) matrix
    @param  phi: R^3
    @return Phi: so(3) matrix
    """
    phi = phi.squeeze()
    Phi = np.array([[0, -phi[2], phi[1]],
                    [phi[2], 0, -phi[0]],
                    [-phi[1], phi[0], 0]])
    return Phi

def adjoint(R):
    """
    Adjoint of SO3 Adjoint (R) = R
    """
    return R

#############################################################################
#                    TODO: Implement your code here                         #
#############################################################################
def motion_model(R, omega, dt):
    """
    @param  R:      State variable (3, 3)
    @param  omega:  gyroscope reading (3,)
    @param  dt:     time step
    @return R_pred: predicted state variable (3, 3)
    """
    R_pred = np.zeros((3,3)) # placeholder
    R_pred = R @ expm(wedge(omega) * dt)
    return R_pred


def measurement_Jacobain(g, R):
    """
    @param  g: gravity (3,)
    @param  R: current pose (3, 3)
    @return H: measurement Jacobain (3, 6)
    """
    H = np.zeros((3,6))
    H[0:3,0:3] = R.T @ wedge(g)
    H[0:3,3:6] = np.eye(3)
    return H

#############################################################################
#                            END OF YOUR CODE                               #
#############################################################################

class imperfect_right_iekf:
    def __init__(self):
        """
        @param system: system and noise models
        """
        self.Phi = np.eye(6)               # state transtion matrix
        self.Q = 1e-3*np.eye(6)            # gyroscope noise covariance
        self.N = 1e-2*np.eye(3)            # accelerometer noise covariance
        self.f = motion_model              # process model
        self.H = measurement_Jacobain      # measurement Jacobain
        self.R = np.eye(3)                 # state robot pose
        self.b = np.zeros(3)               # state accelerometer bias
        self.P = np.eye(6)                 # state covariance

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    def prediction(self, omega, dt):
        """
        @param omega: gyroscope reading
        @param dt:    time step
        """
        adjoint_R_with_I = np.eye(6)
        adjoint_R_with_I[:3, :3] = adjoint(self.R)

        self.R = self.f(self.R, omega, dt)

        self.P = self.Phi @ self.P @ self.Phi.T + adjoint_R_with_I @ (self.Q * dt) @ adjoint_R_with_I.T
        # self.P = self.Phi @ self.P @ self.Phi.T + (self.Q * dt) 

        return

    def correction(self, Y, g):
        """
        @param Y: linear acceleration measurement
        @param g: gravity
        """

        H = self.H(g, self.R)
        N = self.R @ self.N @ self.R.T
        S = H @ self.P @ H.T + N
        L = self.P @ H.T @ np.linalg.inv(S)
        L_R = L[:3]
        L_b = L[3:]

        # Update states
        v = Y - (self.R.T @ g + self.b)
        exp_value_R = expm(wedge(L_R @ v))
        # exp_value_R = expm(wedge(L_R @ (self.R @ Y - g + self.b)))
        self.R = exp_value_R @ self.R
        self.b = self.b + (L_b @ v)
        # self.b = self.b + (L_b @ (self.R @Y - g))

        # Update Covariance
        I = np.eye(6)
        temp = I - L @ H
        self.P = temp @ self.P @ temp.T + L @ N @ L.T

        return

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################


def imperfect_riekf_load_data():
    # load data
    data = {}
    data['accel'] = np.loadtxt('data/a.csv', delimiter=',')
    data['omega'] = np.loadtxt('data/omega.csv', delimiter=',')
    data['dt'] = np.loadtxt('data/dt.csv', delimiter=',')
    data['gravity'] = np.loadtxt('data/gravity.csv', delimiter=',')
    data['euler_gt'] = np.loadtxt('data/euler_gt.csv', delimiter=',')
    return data


def ahrs_imperfect_riekf(inekf_filter, data):
    accel = data['accel']
    omega = data['omega']
    dt = data['dt']
    gravity = data['gravity']
    N = data['accel'].shape[0]
    
    states_rot = np.zeros((N+1, 3, 3))
    states_bias = np.zeros((N+1, 3))
    states_rot[0] = inekf_filter.R
    states_bias[0] = inekf_filter.b
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################
    for i in range(N):

        inekf_filter.prediction(omega[i], dt[i])

        Y = accel[i]
        inekf_filter.correction(Y, gravity)

        states_rot[i+1] = inekf_filter.R
        states_bias[i+1] = inekf_filter.b
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    
    # convert rotation matrices to euler angles
    states_euler = np.zeros((N+1, 3))
    for i, rot in enumerate(states_rot):
        r = R.from_matrix(rot)
        states_euler[i] = r.as_euler('zyx')
    return states_euler, states_bias